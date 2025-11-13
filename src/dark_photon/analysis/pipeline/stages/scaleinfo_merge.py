"""
Scaleinfo merge stage for the analysis pipeline.

This stage:
  * Starts from par-based scaleinfo (ParameterLoadingStage)
  * Merges in TX/RFL fit updates
  * Optionally applies the freqs_from_par override
    (MATLAB AxionAutoRunMain.m / option.freqs_from_par block).
"""

from typing import Any, Dict
import numpy as np

from ..base import PipelineStage, PipelineContext
from ..results import (
    ParameterLoadingResult,
    TransmissionAnalysisResult,
    ReflectionAnalysisResult,
    ScaleinfoMergeResult,
)


class ScaleinfoMergeStage(PipelineStage):
    """
    Stage 4: Merge scaleinfo and apply freqs_from_par overrides.

    Python equivalent of the block after ReadOutCavityTran + readoutbeta in
    AxionAutoRunMain.m:

        if option.freqs_from_par
            scaleinfo.txparams(:,2)  = (Cavity_freq_tx2 + Cavity_freq_tx1)/2;
            scaleinfo.rflparams(:,2) = (Cavity_freq_rfl2 + Cavity_freq_rfl1)/2;
            scaleinfo.freq_beta(:,1) = (Cavity_freq_rfl2 + Cavity_freq_rfl1)/2;

            scaleinfo.rfldriftkHz    = abs(Cavity_freq_rfl1 - Cavity_freq_rfl2)*1e6;
            scaleinfo.txdriftkHz_fit = scaleinfo.txdriftkHz;
            scaleinfo.txdriftkHz     = abs(Cavity_freq_tx1  - Cavity_freq_tx2 )*1e6;
        end
    """

    def execute(self, context: PipelineContext, data: Dict[str, Any]) -> Dict[str, Any]:
        print("  Merging scaleinfo and applying frequency overrides (if enabled).")

        # 1) Get prerequisite stage results
        par_res: ParameterLoadingResult = data.get("parameter_loading")
        tx_res: TransmissionAnalysisResult = data.get("transmission_analysis")
        rfl_res: ReflectionAnalysisResult = data.get("reflection_analysis")

        if par_res is None or not par_res.scaleinfo:
            raise ValueError("ScaleinfoMergeStage requires parameter_loading.scaleinfo")

        # Start from par-based scaleinfo (copy)
        scaleinfo = dict(par_res.scaleinfo)

        # 2) Merge in TX and RFL scaleinfo_updates
        if tx_res is not None and tx_res.scaleinfo_updates:
            for k, v in tx_res.scaleinfo_updates.items():
                scaleinfo[k] = v

        if rfl_res is not None and rfl_res.scaleinfo_updates:
            for k, v in rfl_res.scaleinfo_updates.items():
                scaleinfo[k] = v

        # 3) Apply freqs_from_par override if requested
        if getattr(context.options, "freqs_from_par", False):
            scaleinfo = self._apply_freqs_from_par_override(scaleinfo)

        result = ScaleinfoMergeResult(
            scaleinfo=scaleinfo,
            status="success" if scaleinfo else "failed",
        )

        data["scaleinfo_merge"] = result
        return data

    def _apply_freqs_from_par_override(self, scaleinfo: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implement the MATLAB freqs_from_par override logic on a scaleinfo dict.
        """
        print("    Applying freqs_from_par overrides using par-file frequencies.")

        # Convert lists to numpy arrays for math
        cav_tx1 = np.asarray(scaleinfo["Cavity_freq_tx1"], dtype=float)
        cav_tx2 = np.asarray(scaleinfo["Cavity_freq_tx2"], dtype=float)
        cav_rfl1 = np.asarray(scaleinfo["Cavity_freq_rfl1"], dtype=float)
        cav_rfl2 = np.asarray(scaleinfo["Cavity_freq_rfl2"], dtype=float)

        txparams = np.asarray(scaleinfo["txparams"], dtype=float)   # (N, 5)
        rflparams = np.asarray(scaleinfo["rflparams"], dtype=float) # (N, 5)
        freq_beta = np.asarray(scaleinfo["freq_beta"], dtype=float) # (N, 2)

        # Preserve fit-based drift as txdriftkHz_fit, if present
        if "txdriftkHz" in scaleinfo and "txdriftkHz_fit" not in scaleinfo:
            scaleinfo["txdriftkHz_fit"] = scaleinfo["txdriftkHz"]

        # Override central frequencies (column index 1 -> MATLAB col 2)
        txparams[:, 1] = (cav_tx1 + cav_tx2) / 2.0
        rflparams[:, 1] = (cav_rfl1 + cav_rfl2) / 2.0
        freq_beta[:, 0] = (cav_rfl1 + cav_rfl2) / 2.0

        # Recompute drifts from par frequencies (in kHz)
        rfldrift = np.abs(cav_rfl1 - cav_rfl2) * 1e6
        txdrift = np.abs(cav_tx1 - cav_tx2) * 1e6

        # Write back as lists for consistency
        scaleinfo["txparams"] = txparams.tolist()
        scaleinfo["rflparams"] = rflparams.tolist()
        scaleinfo["freq_beta"] = freq_beta.tolist()
        scaleinfo["rfldriftkHz"] = rfldrift.tolist()
        scaleinfo["txdriftkHz"] = txdrift.tolist()

        return scaleinfo

    def validate_output(self, data: Dict[str, Any]) -> bool:
        """
        Validate merged scaleinfo.
        """
        result: ScaleinfoMergeResult = data.get("scaleinfo_merge")
        if not result or not result.scaleinfo:
            print("  ✗ No merged scaleinfo found")
            return False

        s = result.scaleinfo

        required_fields = ["txparams", "rflparams", "freq_beta", "txdriftkHz", "rfldriftkHz"]
        missing = [f for f in required_fields if f not in s]
        if missing:
            print(f"  ✗ Merged scaleinfo missing fields: {missing}")
            return False

        # Optional consistency check with number of files
        file_enum = data.get("file_enumeration")
        if file_enum and file_enum.files:
            n_files = len(file_enum.files)
            if len(s["txparams"]) != n_files or len(s["rflparams"]) != n_files:
                print(
                    "  ✗ Mismatch between number of files and parameter rows "
                    f"(txparams: {len(s['txparams'])}, rflparams: {len(s['rflparams'])}, files: {n_files})"
                )
                return False

        print("  ✓ Scaleinfo merge and overrides validated")
        return True
