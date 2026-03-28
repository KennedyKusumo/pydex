from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from os import getcwd, makedirs
from pickle import dump
from time import time
from typing import TYPE_CHECKING, Any

import emcee as mc
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    _verbose: int
    _dynamic_system: bool
    _opt_sampling_times: bool
    _n_spt_spec: int
    _bayes_pe_time: float

    n_c: int
    n_spt: int
    n_m_r: int
    n_mp: int
    n_exp: int

    model_parameters: NDArray[np.float64]
    data: NDArray[np.float64] | None
    response: NDArray[np.float64] | None
    residuals: NDArray[np.float64]
    responses_scales: NDArray[np.float64] | None
    measurable_responses: list[int]
    apportionments: NDArray[np.float64] | None
    optimal_candidates: list[Any]
    insilico_data: NDArray[np.float64]
    bayesian_pe_samples: NDArray[np.float64]
    error_cov: NDArray[np.float64]
    error_fim: NDArray[np.float64]
    mp_covar: NDArray[np.float64] | None
    fim: Any


class DesignerEstimation:

    def estimate_parameters(self, bounds: Any, init_guess: NDArray[np.float64] | None = None,
                            method: str = 'trf', update_parameters: bool = False,
                            write: bool = True, options: dict[str, Any] | None = None,
                            max_nfev: int | None = None, variance: float = 1,
                            estimate_covar: bool = True, **kwargs: Any) -> Any:
        from scipy.optimize import least_squares
        if init_guess is None:
            init_guess = self.model_parameters

        if self.data is None:
            raise SyntaxError("No data is put in, do not forget to add it in.")

        if options is None:
            if self._verbose >= 2:
                options = {'disp': True}
            else:
                options = {'disp': False}

        if self._verbose >= 1:
            print("Solving parameter estimation...")
        start = time()

        bounds = np.asarray(bounds)
        bounds = bounds.T
        pe_result = least_squares(
            self._residuals_wrapper_f,
            init_guess,
            bounds=bounds,
            method=method,
            verbose=self._verbose,
            max_nfev=max_nfev,
            **kwargs,
        )

        finish = time()
        if not pe_result.success:
            print('Warning: estimation did not terminate as optimal.')
            stillsave = input(f"Still want to save results? Default is to save results "
                              f"regardless. To skip save, type \"skip\" to terminate: ")
            if stillsave == "skip":
                print("Exiting.")
                return None
            else:
                pass
        print(f"Estimated parameter values:")
        print(np.array2string(
            self.model_parameters,
            separator=","
        ))

        if self._verbose >= 1:
            print(
                "Complete: OLS estimation using %s took %.2f CPU seconds to complete."
                % (
                    method, finish - start))
        if self._verbose >= 2:
            print(
                f"The estimation took a total of {pe_result.nfev} function evaluations"
                f", and {pe_result.njev} number of Jacobian evaluations were done."
            )

        if update_parameters:
            self.model_parameters = pe_result.x
            if self._verbose >= 2:
                print('Nominal parameter value in model updated.')

        if write:
            case_path = getcwd()
            today = datetime.now()
            result_dir = case_path + "/" + str(today.date()) + "_at_" + str(
                today.hour) + "-" + str(
                today.minute) + "-" + str(today.second)
            makedirs(result_dir)
            with open(result_dir + "/result_file.pkl", "wb") as file:
                dump(pe_result, file)
            if self._verbose >= 2:
                print('Parameter estimation result saved to: %s.' % result_dir)

        if estimate_covar:
            try:
                self.eval_fim(
                    efforts=np.ones((self.n_c, self.n_spt)),
                )
            except RuntimeWarning:
                print(
                    f"Sensitivity analysis for computing the information matrix failed, "
                    f"leading to a runtime error. Skipping the estimation of the covariance "
                    f"matrix."
                )
                return pe_result
            try:
                self.mp_covar = variance * np.linalg.inv(self.fim)
            except np.linalg.LinAlgError:
                try:
                    self.mp_covar = variance * np.linalg.pinv(self.fim)
                except np.linalg.LinAlgError:
                    return pe_result

            if self.mp_covar is not None and self._verbose >= 1:
                print(f"Standard absolute error of estimates (noise variance = {variance}):")
                for p, mp in enumerate(self.model_parameters):
                    print(fr"{mp:>40} +- {np.sqrt(np.diag(self.mp_covar)[p])}")
                print(f"Standard relative error of estimates (noise variance = {variance}):")
                for p, mp in enumerate(self.model_parameters):
                    print(fr"{mp:>40} +- {np.sqrt(np.diag(self.mp_covar)[p]) / mp * 100} %")

        return pe_result

    def estimate_parameters_alt(self, init_guess, bounds, method='l-bfgs-b',
                                update_parameters=False, write=True, options=None,
                                variance=1, **kwargs):
        from scipy.optimize import minimize
        if self.data is None:
            raise SyntaxError("No data is put in, do not forget to add it in.")

        if options is None:
            if self._verbose >= 2:
                options = {'disp': True}
            else:
                options = {'disp': False}

        if self._verbose >= 1:
            print("Solving parameter estimation...")
        start = time()

        pe_result = minimize(
            self._residuals_wrapper_f_old,
            init_guess,
            bounds=bounds,
            method=method,
            options=options,
            **kwargs,
        )
        finish = time()
        if not pe_result.success:
            print('Fail: estimation did not converge; exiting.')
            return None
        print(f"Estimated parameter values:")
        print(np.array2string(
            self.model_parameters,
            separator=","
        ))

        if self._verbose >= 1:
            print(
                "Complete: OLS estimation using %s took %.2f CPU seconds to complete."
                % (
                    method, finish - start))
        if self._verbose >= 2:
            print(
                "The estimation took a total of %d function evaluations, %d used for "
                "numerical estimation of the "
                "Jacobian using forward finite differences." % (
                    pe_result.nfev, pe_result.nfev - pe_result.nit - 1))

        if update_parameters:
            self.model_parameters = pe_result.x
            if self._verbose >= 2:
                print('Nominal parameter value in model updated.')

        if write:
            case_path = getcwd()
            today = datetime.now()
            result_dir = case_path + "/" + str(today.date()) + "_at_" + str(
                today.hour) + "-" + str(
                today.minute) + "-" + str(today.second)
            makedirs(result_dir)
            with open(result_dir + "/result_file.pkl", "wb") as file:
                dump(pe_result, file)
            if self._verbose >= 2:
                print('Parameter estimation result saved to: %s.' % result_dir)

        try:
            self.eval_fim(
                efforts=np.ones((self.n_c, self.n_spt)),
                mp=pe_result.x,
            )
        except RuntimeWarning:
            print(
                f"Sensitivity analysis for computing the information matrix failed, "
                f"leading to a runtime error. Skipping the estimation of the covariance "
                f"matrix."
            )
            return pe_result
        try:
            self.mp_covar = np.linalg.inv(self.fim)
        except np.linalg.LinAlgError:
            try:
                self.mp_covar = np.linalg.pinv(self.fim)
            except np.linalg.LinAlgError:
                return pe_result

        if self.mp_covar is not None:
            print(f"Standard absolute error of estimates (noise variance = {variance}):")
            for p, mp in enumerate(self.model_parameters):
                print(fr"{mp:>40} +- {np.sqrt(np.diag(self.mp_covar)[p])}")
            print(f"Standard relative error of estimates (noise variance = {variance}):")
            for p, mp in enumerate(self.model_parameters):
                print(fr"{mp:>40} +- {np.sqrt(np.diag(self.mp_covar)[p]) / mp * 100} %")

        return pe_result

    def insilico_bayesian_inference(self, n_walkers: int, n_steps: int, burn_in: int,
                                    verbose: bool = True,
                                    prior_pdf: Callable[..., float] | None = None,
                                    bounds: Any = None, seed: int = 123456,
                                    write: bool = True) -> Any:
        if self._verbose >= 1:
            print(f"".center(100, "="))
        np.random.seed(seed)

        if prior_pdf is None:
            if bounds is None:
                raise SyntaxWarning(
                    "Please provide either the prior_pdf function or bounds, Pydex "
                    "assumes uniform distribution between the given bounds."
                )
            else:
                prior_pdf = self.uniform_prior_pdf(bounds)

        def likelihood_function(p):
            lkhd = 0
            for c, (ti, tv, sp) in enumerate(zip(tic, tvc, spt)):
                y_pred = self._simulate_internal(ti, tv, p, sp)
                delta = y_pred - data[c]
                for j in range(self._n_spt_spec):
                    lkhd += np.log(1 / np.sqrt((2 * np.pi) ** self.n_mp * np.linalg.det(self.error_cov)) * np.exp(-1/2 * delta[j] @ self.error_fim @ delta[j].T))
            return lkhd

        def log_prob(p):
            return likelihood_function(p) + prior_pdf(p)

        self._bayes_pe_time = time()
        sampler = mc.EnsembleSampler(
            nwalkers=n_walkers,
            ndim=self.n_mp,
            log_prob_fn=log_prob,
        )
        init_pos = np.random.uniform(0.95, 1.05, size=(n_walkers, self.n_mp)) * self.model_parameters
        sampler.run_mcmc(
            init_pos,
            nsteps=n_steps,
            progress=verbose,
        )
        tau = sampler.get_autocorr_time()
        self._bayes_pe_time = time() - self._bayes_pe_time
        try:
            thin_factor = int(np.round(np.nanmean(tau) / 2))
            self.bayesian_pe_samples = sampler.get_chain(discard=burn_in, thin=thin_factor, flat=True)
        except ValueError:
            self.bayesian_pe_samples = sampler.get_chain(discard=burn_in, flat=True)
            print(
                f"[WARNING]: ValueError when computing auto-correlation time, "
                f"Pydex did not specify any kwarg when getting the chain from emcee. "
            )

        if self._verbose >= 3:
            print(self.bayesian_pe_samples.shape)
        if self._verbose >= 1:
            print(
                f"In-silico Bayesian Inference for {self.n_exp} of experiments completed "
                f"within {self._bayes_pe_time:.2f} seconds.",
            )
            print(f"".center(100, "="))

        if write:
            fn = f"insilico_bayes_pe_samples_{self.n_exp}_exp_{n_walkers}_walkers_{n_steps}_steps_{burn_in}_burnin_{seed}_seed"
            fp = self._generate_result_path(fn, "pkl")
            dump(self.bayesian_pe_samples, open(fp, 'wb'))

        return self.bayesian_pe_samples

    def insilico_bayesian_inference(self, n_walkers, n_steps, burn_in, verbose=True, prior_pdf=None, bounds=None, seed=123456, write=True):
        self.insilico_data = self.generate_insilico_data(seed)
        tic, tvc, spt = self._get_apportioned_candidates()
        return self.bayesian_inference(tic, tvc, spt, self.insilico_data, n_walkers, n_steps, burn_in, verbose, prior_pdf, bounds, seed, write)

    def generate_insilico_data(self, seed=123456):
        if self.apportionments is None:
            raise SyntaxWarning(
                "Please run an apportionment before running an in-silico activity with "
                "Pydex."
            )
        np.random.seed(seed)
        tic, tvc, spt = self._get_apportioned_candidates()
        if self._opt_sampling_times:
            self.insilico_data = np.empty((self.n_exp, self._n_spt_spec, self.n_m_r))
            for c, (ti, tv, sp) in enumerate(zip(tic, tvc, spt)):
                self.insilico_data[c] = self._simulate_internal(ti, tv, self.model_parameters, sp)
            self.insilico_data += np.random.multivariate_normal(
                np.zeros(self.n_m_r),
                cov=self.error_cov,
                size=(self.n_exp, self._n_spt_spec),
            )
        else:
            self.insilico_data = np.empty((self.n_exp, self.n_spt, self.n_m_r))
            for c, (ti, tv, sp) in enumerate(zip(tic, tvc, spt)):
                self.insilico_data[c] = self._simulate_internal(ti, tv, self.model_parameters, sp)
            self.insilico_data += np.random.multivariate_normal(
                np.zeros(self.n_m_r),
                cov=self.error_cov,
                size=(self.n_exp, self.n_spt),
            )
        return self.insilico_data

    def eval_residuals(self, model_parameters):
        self.model_parameters = model_parameters

        """ run the model to get predictions """
        self.simulate_candidates()
        if self._dynamic_system:
            self.residuals = self.data - self.response[:, :, self.measurable_responses]
        else:
            self.residuals = self.data - self.response[:, self.measurable_responses]

        return self.residuals[
            ~np.isnan(self.residuals)]  # return residuals where entries are not empty

    def uniform_prior_pdf(self, bounds):
        def prior_f(p):
            out: float = 0
            for i, bound in enumerate(bounds):
                if bound[0] <= p[i] <= bound[1]:
                    out += 0
                else:
                    out -= np.inf
            return out
        return prior_f

    def _get_apportioned_candidates(self):
        app_tic_candidates = []
        app_tvc_candidates = []
        app_spt_candidates = []
        for i, app in enumerate(self.apportionments):
            tic = self.optimal_candidates[i][1]
            tvc = self.optimal_candidates[i][2]
            spt = self.optimal_candidates[i][3]
            for _ in range(int(app)):
                app_tic_candidates.append(tic)
                app_tvc_candidates.append(tvc)
                app_spt_candidates.append(spt)
        app_tic_candidates = np.array(app_tic_candidates)
        app_tvc_candidates = np.array(app_tvc_candidates)
        app_spt_candidates = np.array(app_spt_candidates)
        return app_tic_candidates, app_tvc_candidates, app_spt_candidates

    def _residuals_wrapper_f(self, model_parameters):
        if self.responses_scales is None:
            if self._dynamic_system:
                self.responses_scales = np.nanmean(self.data, axis=(0, 1))
            else:
                self.responses_scales = np.nanmean(self.data, axis=0)

        self.eval_residuals(model_parameters)
        if self._dynamic_system:
            res = self.residuals / self.responses_scales[None, None, :]
        else:
            res = self.residuals / self.responses_scales[None, :]
        if self._dynamic_system:
            res = res.reshape(self.n_c * self.n_spt, self.n_m_r)
        else:
            res = res.reshape(self.n_c, self.n_m_r)
        return res[~np.isnan(res)]

    def _residuals_wrapper_f_old(self, model_parameters):
        if self.responses_scales is None:
            if self._dynamic_system:
                self.responses_scales = np.nanmean(self.data, axis=(0, 1))
            else:
                self.responses_scales = np.nanmean(self.data, axis=0)
        self.eval_residuals(model_parameters)
        res = self.residuals / self.responses_scales[None, None, :]
        res = res[~np.isnan(res)]
        return np.squeeze(res[None, :] @ res[:, None])
