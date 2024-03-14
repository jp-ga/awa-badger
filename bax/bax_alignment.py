from xopt.generators.bayesian.bax.algorithms import Algorithm
from abc import ABC
from pydantic import Field
from scipy.optimize import minimize
from torch import Tensor
from botorch.models.model import Model, ModelList
from typing import Dict, Tuple, Union
import torch

def unif_random_sample_domain(n_samples, domain):
    ndim = len(domain)

    # uniform sample, rescaled, and shifted to cover the domain
    x_samples = torch.rand(n_samples, ndim) * torch.tensor(
        [bounds[1] - bounds[0] for bounds in domain]
    ) + torch.tensor([bounds[0] for bounds in domain])

    return x_samples

class ScipyBeamAlignment(Algorithm, ABC):
    name = "ScipyBeamAlignment"
    meas_dims: Union[int, list[int]] = Field(
        description="list of indeces identifying the measurement quad dimensions in the model"
    )
    x_key: str = Field(None,
        description="oberservable name for x centroid position"
    )
    y_key: str = Field(None,
        description="oberservable name for y centroid position"
    )

    @property
    def observable_names_ordered(self) -> list:  
        # get observable model names in the order they appear in the model (ModelList)
        return [key for key in [self.x_key, self.y_key] if key]
    
    def get_execution_paths(
        self, model: ModelList, bounds: Tensor, verbose=False
    ) -> Tuple[Tensor, Tensor, Dict]:
        """get execution paths that minimize the objective function"""

        meas_scans = torch.index_select(
            bounds.T, dim=0, index=torch.tensor(self.meas_dims)
        )
        ndim = bounds.shape[1]
        tuning_dims = [i for i in range(ndim) if i not in self.meas_dims]
        tuning_domain = torch.index_select(
            bounds.T, dim=0, index=torch.tensor(tuning_dims)
        )

        device = torch.tensor(1).device
        torch.set_default_tensor_type("torch.DoubleTensor")

        cpu_models = [copy.deepcopy(m).cpu() for m in model.models]
        sample_funcs_list = [
            draw_linear_product_kernel_post_paths(cpu_model, n_samples=self.n_samples)
                for cpu_model in cpu_models
            ]

        xs_tuning_init = unif_random_sample_domain(
            self.n_samples, tuning_domain
        ).double()

        x_tuning_init = xs_tuning_init.flatten()

        # minimize
        def target_func_for_scipy(x_tuning_flat):
            return (
                self.sum_samplewise_misalignment_flat_x(
                    sample_funcs_list,
                    torch.tensor(x_tuning_flat),
                    self.meas_dims,
                    meas_scans.cpu(),
                )
                .detach()
                .cpu()
                .numpy()
            )

        def target_func_for_torch(x_tuning_flat):
            return self.sum_samplewise_misalignment_flat_x(
                sample_funcs_list, x_tuning_flat, self.meas_dims, meas_scans.cpu()
            )

        def target_jac(x):
            return (
                torch.autograd.functional.jacobian(
                    target_func_for_torch, torch.tensor(x)
                )
                .detach()
                .cpu()
                .numpy()
            )

        res = minimize(
            target_func_for_scipy,
            x_tuning_init.detach().cpu().numpy(),
            jac=target_jac,
            bounds=tuning_domain.repeat(self.n_samples, 1).detach().cpu().numpy(),
            options={"eps": 1e-03},
        )
        if verbose:
            print(
                "ScipyBeamAlignment evaluated",
                self.n_samples,
                "(pathwise) posterior samples",
                res.nfev,
                "times in get_sample_optimal_tuning_configs().",
            )

            print(
                "ScipyBeamAlignment evaluated",
                self.n_samples,
                "(pathwise) posterior sample jacobians",
                res.njev,
                "times in get_sample_optimal_tuning_configs().",
            )

            print(
                "ScipyBeamAlignment took",
                res.nit,
                "steps in get_sample_optimal_tuning_configs().",
            )

        x_tuning_best_flat = torch.tensor(res.x)

        x_tuning_best = x_tuning_best_flat.reshape(
            self.n_samples, 1, -1
        )  # each row represents its respective sample's optimal tuning config


        if device.type == "cuda":
            torch.set_default_tensor_type("torch.cuda.DoubleTensor")

        xs = self.get_meas_scan_inputs(x_tuning_best, meas_scans, self.meas_dims)
        xs_exe = xs
        
        # evaluate posterior samples at input locations
        ys_exe_list = [sample_func(xs_exe).reshape(
            self.n_samples, 1+len(self.meas_dims), 1
        ) for sample_func in sample_funcs_list]
        ys_exe = torch.cat(ys_exe_list, dim=-1)
                            
        results_dict = {
            "xs_exe": xs_exe,
            "ys_exe": ys_exe,
            "x_tuning_best": x_tuning_best,
            "sample_funcs_list": sample_funcs_list,
        }

        return xs_exe, ys_exe, results_dict
        
    def sample_funcs_misalignment(
        self,
        sample_funcs_list,
        x_tuning,  # n x d tensor
        meas_dims,  # list of integers
        meas_scans,  # tensor of measurement device(s) scan inputs, shape: len(meas_dims) x 2
    ):
        """
        A function that computes the beam misalignment(s) through a set of measurement quadrupoles
        from a set of pathwise samples taken from a SingleTaskGP model of the beam centroid position with
        respect to some tuning devices and some measurement quadrupoles.

        arguments:
            sample_funcs_list: a list of pathwise posterior samples for x, y, or both 
                        from a SingleTaskGP model of the beam centroid positions (assumes Linear ProductKernel)
            x_tuning: a tensor of shape (n_samples x n_tuning_dims) where the nth row defines a point in
                        tuning-parameter space at which to evaluate the misalignment of the nth
                        posterior pathwise sample given by post_paths
            meas_dims: the dimension indeces of our model that describe the quadrupole measurement devices
            meas_scans: a tensor of measurement scan inputs, shape len(meas_dims) x 2, where the nth row
                        contains two input scan values for the nth measurement quadrupole

         returns:
             misalignment: the sum of the squared slopes of the beam centroid model output with respect to the
                             measurement quads
             xs: the virtual scan inputs
             ys: the virtual scan outputs (beam centroid positions)

        NOTE: meas scans only needs to have 2 values for each device because it is expected that post_paths
                are produced from a SingleTaskGP with Linear ProductKernel (i.e. post_paths should have
                linear output for each dimension).
        """
        xs = self.get_meas_scan_inputs(x_tuning, meas_scans, meas_dims)

        sample_misalignments_sum_list = [] # list to store the sum of the samplewise misalignments in x, y or both
        sample_ys_list = [] # list to store the centroid positions for x, y or both
        for sample_func in sample_funcs_list:
            ys = sample_func(xs)
            ys = ys.reshape(self.n_samples, -1)

            rise = ys[:, 1:] - ys[:, 0].reshape(-1, 1)
            run = (meas_scans[:, 1] - meas_scans[:, 0]).T.repeat(ys.shape[0], 1)
            slope = rise / run

            misalignment = slope.pow(2).sum(dim=1)
            sample_misalignments_sum_list += [misalignment]
            sample_ys_list += [ys]
        
        total_misalignment = sum(sample_misalignments_sum_list)
        return total_misalignment, xs, sample_ys_list

    def get_meas_scan_inputs(self, x_tuning, meas_scans, meas_dims):
        # meas_scans = torch.index_select(
        #     bounds.T, dim=0, index=torch.tensor(self.meas_dims)
        # )    
        n_steps_meas_scan = 1 + len(meas_dims)
        n_tuning_configs = x_tuning.shape[0]

        # construct measurement scan inputs
        xs = torch.repeat_interleave(x_tuning, n_steps_meas_scan, dim=-2)

        for i in range(len(meas_dims)):
            meas_dim = meas_dims[i]
            meas_scan = meas_scans[i]
            full_scan_column = meas_scan[0].repeat(n_steps_meas_scan, 1)
            full_scan_column[i + 1, 0] = meas_scan[1]
            full_scan_column_repeated = full_scan_column.repeat(*x_tuning.shape[:-1], 1)

            xs = torch.cat(
                (xs[..., :meas_dim], full_scan_column_repeated, xs[..., meas_dim:]), dim=-1
            )

        return xs

    def sum_samplewise_misalignment_flat_x(
        self, sample_funcs_list, x_tuning_flat, meas_dims, meas_scans
    ):
        """
        A wrapper function that computes the sum of the samplewise misalignments for more convenient
        minimization with scipy.

        arguments:
            Same as post_path_misalignment() EXCEPT:

            x_tuning_flat: a FLATTENED tensor formerly of shape (n_samples x ndim) where the nth
                            row defines a point in tuning-parameter space at which to evaluate the
                            misalignment of the nth posterior pathwise sample given by post_paths

            NOTE: x_tuning_flat must be 1d (flattened) so the output of this function can be minimized
                    with scipy minimization routines (that expect a 1d vector of inputs)
            NOTE: samplewise is set to True to avoid unncessary computation during simultaneous minimization
                    of the pathwise misalignments.
        """

        x_tuning = x_tuning_flat.double().reshape(self.n_samples, 1, -1)

        return torch.sum(
            self.sample_funcs_misalignment(
                sample_funcs_list, x_tuning, meas_dims, meas_scans
            )[0]
        )


import copy
import math

def draw_poly_kernel_prior_paths(
    poly_kernel, n_samples
):  # poly_kernel is a scaled polynomial kernel
    c = poly_kernel.offset
    degree = poly_kernel.power
    ws = torch.randn(size=[n_samples, 1, degree + 1], device=c.device)

    def paths(xs):
        if (
            len(xs.shape) == 2 and xs.shape[1] == 1
        ):  # xs must be n_samples x npoints x 1 dim
            xs = xs.repeat(n_samples, 1, 1)  # duplicate over batch (sample) dim

        coeffs = [math.comb(degree, i) for i in range(degree + 1)]
        X = torch.concat(
            [
                (coeff * c.pow(i)).sqrt() * xs.pow(degree - i)
                for i, coeff in enumerate(coeffs)
            ],
            dim=2,
        )
        W = ws.repeat(1, xs.shape[1], 1)  # ws is n_samples x 1 x 3 dim

        phis = W * X
        return torch.sum(phis, dim=-1)  # result tensor is shape n_samples x npoints

    return paths

def draw_linear_product_kernel_prior_paths(model, n_samples):
    ndim = model.train_inputs[0].shape[1]

    outputscale = copy.copy(model.covar_module.outputscale.detach())
    kernels = []
    dims = []

    for i in range(len(model.covar_module.base_kernel.kernels)):
        lin_kernel = copy.deepcopy(model.covar_module.base_kernel.kernels[i])
        kernels += [lin_kernel]
        dims += [lin_kernel.active_dims]

    lin_prior_paths = [
        draw_poly_kernel_prior_paths(kernel, n_samples) for kernel in kernels
    ]

    def linear_product_kernel_prior_paths(xs):
        ys_lin = []
        for i in range(len(lin_prior_paths)):
            xs_lin = torch.index_select(xs, dim=-1, index=dims[i]).float()
            ys_lin += [lin_prior_paths[i](xs_lin)]
        output = 1.0
        for ys in ys_lin:
            output *= ys
        return (outputscale.sqrt() * output).double()

    return linear_product_kernel_prior_paths

def draw_linear_product_kernel_post_paths(model, n_samples, cpu=True):
    linear_product_kernel_prior_paths = draw_linear_product_kernel_prior_paths(
        model, n_samples=n_samples
    )

    train_x = model.train_inputs[0]

    train_y = model.train_targets.reshape(-1, 1)

    train_y = train_y - model.mean_module(train_x).reshape(train_y.shape)

    Knn = model.covar_module(train_x, train_x)

    sigma = torch.sqrt(model.likelihood.noise[0])

    K = Knn + sigma**2 * torch.eye(Knn.shape[0])

    prior_residual = train_y.repeat(n_samples, 1, 1).reshape(
        n_samples, -1
    ) - linear_product_kernel_prior_paths(train_x)
    prior_residual -= sigma * torch.randn_like(prior_residual)

    Lnn = torch.cholesky(K.to_dense())
    batched_lnn = torch.stack([Lnn] * n_samples)
    batched_lnnt = torch.stack([Lnn.T] * n_samples)

    vbar = torch.linalg.solve(batched_lnn, prior_residual)
    v = torch.linalg.solve(batched_lnnt, vbar)
    v = v.reshape(-1, 1)

    v = v.reshape(n_samples, -1, 1)
    v_t = v.transpose(1, 2)

    def post_paths(xs):
        if model.input_transform is not None:
            xs = model.input_transform(xs)

        K_update = model.covar_module(train_x, xs.double())

        update = torch.matmul(v_t, K_update)
        update = update.reshape(n_samples, -1)

        prior = linear_product_kernel_prior_paths(xs)

        post = prior + update + model.mean_module(xs)
        if model.outcome_transform is not None:
            post = model.outcome_transform.untransform(post)[0]

        return post

    post_paths.n_samples = n_samples

    return post_paths