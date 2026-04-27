# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/tests/sm_correlation.py                        ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulation Manager — Análise de correlação e ensemble     ║
# ║  Criação     : 2026-04-26                                                 ║
# ║  Status      : Produção (v2.6)                                            ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Funções e dialogs para análise estatística do tensor H:                ║
# ║      • Matriz de correlação Pearson/Spearman/Kendall entre componentes    ║
# ║      • Envelope P5/P95 do ensemble de modelos                             ║
# ║      • Detecção de outliers via z-score                                   ║
# ║                                                                           ║
# ║  USO                                                                      ║
# ║    from sm_correlation import compute_correlation_matrix                  ║
# ║    matrix, labels = compute_correlation_matrix(H_stack, method='kendall')║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Análise de correlação e ensemble para v2.6 (P3)."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "compute_correlation_matrix",
    "compute_ensemble_envelope",
    "detect_outliers_zscore",
    "EM_COMPONENT_LABELS",
    "CorrelationAnalysisDialog",
    "EnsembleAnalysisDialog",
]

EM_COMPONENT_LABELS: List[str] = [
    "Hxx",
    "Hxy",
    "Hxz",
    "Hyx",
    "Hyy",
    "Hyz",
    "Hzx",
    "Hzy",
    "Hzz",
]
GEOSIGNAL_LABELS: List[str] = ["USD", "UAD", "UHR", "UHA", "U3DF"]


def compute_correlation_matrix(
    H_stack: np.ndarray,
    method: str = "pearson",
    component_indices: Optional[List[int]] = None,
    include_geosignals: bool = False,
    use_real_part: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """Computa matriz NxN de correlação entre componentes EM ao longo de z_obs.

    Args:
        H_stack: Tensor com shape ``(nTR, nAng, n_pos, nf, 9)`` (sem ensemble)
            ou ``(n_models, nTR, nAng, n_pos, nf, 9)`` (ensemble agrega).
        method: ``"pearson"`` (default), ``"spearman"`` ou ``"kendall"``.
        component_indices: subset de componentes (índices 0..8). None = todos.
        include_geosignals: se True, anexa USD/UAD/UHR/UHA aos componentes EM
            (currently no-op stub — reservado para v2.6.x).
        use_real_part: se True usa Re(H), senão usa |H|. Default True.

    Returns:
        ``(matrix, labels)`` onde:
          - ``matrix`` é (N, N) com valores em [-1, 1]
          - ``labels`` lista de N nomes (e.g., ['Hxx', 'Hzz', ...])

    Raises:
        ValueError: se method não reconhecido ou tensor com shape inválido.

    Note:
        Para ensemble (6D), agrega via mean sobre n_models antes de correlacionar.
        Cada combinação (iTR, iAng, ifq) gera uma série temporal ao longo de
        n_pos; correlações são calculadas e médias entre combinações.
    """
    valid_methods = {"pearson", "spearman", "kendall"}
    if method not in valid_methods:
        raise ValueError(f"method deve ser um de {valid_methods}, recebeu: {method!r}")
    H = np.asarray(H_stack)
    if H.ndim == 6:
        # Ensemble: agrega via média antes de correlacionar
        H = H.mean(axis=0)
    if H.ndim != 5:
        raise ValueError(
            f"H_stack deve ter shape 5D (nTR,nAng,n_pos,nf,9) ou 6D com "
            f"ensemble, recebeu shape {H.shape}"
        )
    nTR, nAng, n_pos, nf, n_comp = H.shape
    if n_comp != 9:
        raise ValueError(f"Última dimensão deve ser 9 (componentes EM), recebeu {n_comp}")

    if component_indices is None:
        component_indices = list(range(9))
    indices = list(component_indices)
    n = len(indices)
    labels = [EM_COMPONENT_LABELS[i] for i in indices]

    # Extrai parte real ou magnitude
    if use_real_part:
        data = H.real
    else:
        data = np.abs(H)

    # Para cada combinação (itr, iang, ifq), calcula matriz NxN, depois média
    accum = np.zeros((n, n), dtype=np.float64)
    counts = 0
    for itr in range(nTR):
        for iang in range(nAng):
            for ifq in range(nf):
                series_matrix = data[itr, iang, :, ifq, :][:, indices]  # (n_pos, n)
                m = _corrmat(series_matrix, method)
                accum += m
                counts += 1
    if counts == 0:
        return np.eye(n), labels
    matrix = accum / counts
    return matrix, labels


def _corrmat(data: np.ndarray, method: str) -> np.ndarray:
    """Helper interno: matriz NxN para um único array (n_obs, n_features)."""
    n_features = data.shape[1]
    if method == "pearson":
        # NumPy nativo é vetorizado e rápido
        if data.shape[0] < 2:
            return np.eye(n_features)
        with np.errstate(invalid="ignore", divide="ignore"):
            mat = np.corrcoef(data, rowvar=False)
        return np.nan_to_num(mat, nan=0.0)
    # Spearman e Kendall via scipy
    try:
        from scipy.stats import kendalltau, spearmanr
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "compute_correlation_matrix(method='spearman'|'kendall') requer scipy"
        ) from exc
    mat = np.eye(n_features)
    for i in range(n_features):
        for j in range(i + 1, n_features):
            xi = data[:, i]
            xj = data[:, j]
            if method == "spearman":
                r, _ = spearmanr(xi, xj)
            else:  # kendall
                r, _ = kendalltau(xi, xj)
            r = float(r) if np.isfinite(r) else 0.0
            mat[i, j] = r
            mat[j, i] = r
    return mat


def compute_ensemble_envelope(
    H_stack: np.ndarray,
    percentiles: Tuple[float, float] = (5.0, 95.0),
) -> Dict[str, np.ndarray]:
    """Mediana + envelope percentil do ensemble.

    Args:
        H_stack: Tensor 6D com shape ``(n_models, nTR, nAng, n_pos, nf, 9)``.
        percentiles: tuple ``(p_low, p_high)`` em porcentos. Default (5, 95).

    Returns:
        Dict com chaves ``"median"``, ``"p_low"``, ``"p_high"``, todas com
        shape ``(nTR, nAng, n_pos, nf, 9)``.

    Raises:
        ValueError: se H_stack não for 6D.
    """
    H = np.asarray(H_stack)
    if H.ndim != 6:
        raise ValueError(
            f"compute_ensemble_envelope requer shape 6D (n_models, ...), "
            f"recebeu {H.shape}"
        )
    p_low, p_high = float(percentiles[0]), float(percentiles[1])
    return {
        "median": np.median(H, axis=0),
        "p_low": np.percentile(H, p_low, axis=0),
        "p_high": np.percentile(H, p_high, axis=0),
    }


def detect_outliers_zscore(
    H_stack: np.ndarray,
    threshold: float = 3.0,
    component_index: int = 8,  # Hzz default
) -> np.ndarray:
    """Detecta modelos outliers via z-score por posição.

    Args:
        H_stack: Tensor 6D ``(n_models, nTR, nAng, n_pos, nf, 9)``.
        threshold: z-score absoluto considerado outlier (default 3.0).
        component_index: componente EM a inspecionar (default 8 = Hzz).

    Returns:
        Array bool de shape ``(n_models,)`` — True para modelos outliers
        (z-score > threshold em mais de 10% das posições).
    """
    H = np.asarray(H_stack)
    if H.ndim != 6:
        raise ValueError(f"Requer shape 6D, recebeu {H.shape}")
    # Extrai componente, agrega TR/ang/freq via mean
    comp = np.abs(H[..., component_index]).mean(axis=(1, 2, 4))  # (n_models, n_pos)
    mu = comp.mean(axis=0)
    sigma = comp.std(axis=0)
    sigma = np.where(sigma > 0, sigma, 1e-12)
    z = np.abs(comp - mu[None, :]) / sigma[None, :]
    frac_above = (z > threshold).mean(axis=1)
    return frac_above > 0.1


# ══════════════════════════════════════════════════════════════════════════
# v2.6b L5 — Dialogs Qt para análise interativa do ensemble
# ══════════════════════════════════════════════════════════════════════════

try:
    from PyQt6 import QtCore, QtGui, QtWidgets

    _QT_OK = True
except ImportError:
    try:
        from PyQt5 import QtCore, QtGui, QtWidgets  # type: ignore

        _QT_OK = True
    except ImportError:
        try:
            from PySide6 import QtCore, QtGui, QtWidgets  # type: ignore

            _QT_OK = True
        except ImportError:
            _QT_OK = False


if _QT_OK:

    class CorrelationAnalysisDialog(QtWidgets.QDialog):
        """Dialog interativo para matriz de correlação entre componentes EM.

        UI:
          • Top: combo método (pearson/spearman/kendall) + botão "Recomputar"
          • Center: heatmap matplotlib (imshow + colorbar divergente)
          • Bottom: tabela com valores numéricos
          • Click em célula → highlight + valor exibido em label

        Note:
            Para tensores grandes (n_pos > 5000), Spearman e Kendall podem
            demorar — o cálculo é feito on-demand quando o usuário clica
            em "Recomputar" (default Pearson).
        """

        def __init__(
            self,
            parent: Optional[QtWidgets.QWidget],
            H_stack: np.ndarray,
            *,
            default_method: str = "pearson",
        ) -> None:
            super().__init__(parent)
            self.setWindowTitle("Análise — Matriz de Correlação dos Componentes EM")
            self.setModal(False)
            self.resize(820, 720)
            self._H = np.asarray(H_stack)
            self._matrix: Optional[np.ndarray] = None
            self._labels: List[str] = list(EM_COMPONENT_LABELS)

            # Top bar: método + recomputar
            top = QtWidgets.QHBoxLayout()
            top.addWidget(QtWidgets.QLabel("Método:"))
            self.combo_method = QtWidgets.QComboBox()
            self.combo_method.addItems(["pearson", "spearman", "kendall"])
            self.combo_method.setCurrentText(default_method)
            top.addWidget(self.combo_method)
            self.check_real = QtWidgets.QCheckBox("Usar Re(H) (senão |H|)")
            self.check_real.setChecked(True)
            top.addWidget(self.check_real)
            self.btn_recompute = QtWidgets.QPushButton("Recomputar")
            self.btn_recompute.clicked.connect(self._on_recompute)
            top.addWidget(self.btn_recompute)
            top.addStretch(1)

            # Lazy import matplotlib canvas — usa o EMCanvas do projeto
            try:
                from .sm_plots import EMCanvas, PlotStyle

                self.canvas = EMCanvas(self, figsize=(7, 5), style=PlotStyle())
            except Exception:
                self.canvas = QtWidgets.QLabel("Matplotlib indisponível.")

            # Tabela de valores
            self.table = QtWidgets.QTableWidget(0, 0)
            self.table.setMinimumHeight(180)
            self.table.itemDoubleClicked.connect(self._on_cell_double_clicked)

            # Status label
            self.lbl_status = QtWidgets.QLabel("Pronto.")
            self.lbl_status.setStyleSheet("color:#888888; font-size:11px;")

            # Layout raiz
            root = QtWidgets.QVBoxLayout(self)
            root.addLayout(top)
            root.addWidget(self.canvas, 1)
            root.addWidget(self.table)
            root.addWidget(self.lbl_status)

            close_btn = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.StandardButton.Close
            )
            close_btn.rejected.connect(self.reject)
            root.addWidget(close_btn)

            # Computa matriz inicial
            self._on_recompute()

        def _on_recompute(self) -> None:
            method = self.combo_method.currentText()
            use_real = self.check_real.isChecked()
            try:
                self.lbl_status.setText(f"Computando matriz {method}…")
                QtWidgets.QApplication.processEvents()
                self._matrix, self._labels = compute_correlation_matrix(
                    self._H, method=method, use_real_part=use_real
                )
                self._refresh_heatmap()
                self._refresh_table()
                self.lbl_status.setText(
                    f"OK — método: {method}, "
                    f"shape: {self._matrix.shape}, "
                    f"min/max: {float(self._matrix.min()):.3f} / {float(self._matrix.max()):.3f}"
                )
            except Exception as exc:
                self.lbl_status.setText(f"Erro: {exc}")

        def _refresh_heatmap(self) -> None:
            if self._matrix is None or not hasattr(self.canvas, "figure"):
                return
            fig = self.canvas.figure
            fig.clear()
            ax = fig.add_subplot(111)
            im = ax.imshow(
                self._matrix,
                cmap="RdBu_r",
                vmin=-1.0,
                vmax=1.0,
                interpolation="nearest",
            )
            ax.set_xticks(range(len(self._labels)))
            ax.set_yticks(range(len(self._labels)))
            ax.set_xticklabels(self._labels, rotation=45, ha="right")
            ax.set_yticklabels(self._labels)
            ax.set_title(f"Correlação ({self.combo_method.currentText()})")
            # Anota valores nas células
            for i in range(len(self._labels)):
                for j in range(len(self._labels)):
                    val = float(self._matrix[i, j])
                    color = "white" if abs(val) > 0.5 else "black"
                    ax.text(
                        j,
                        i,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=8,
                    )
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            try:
                self.canvas.canvas.draw_idle()
            except Exception:
                pass

        def _refresh_table(self) -> None:
            if self._matrix is None:
                return
            n = len(self._labels)
            self.table.setRowCount(n)
            self.table.setColumnCount(n)
            self.table.setHorizontalHeaderLabels(self._labels)
            self.table.setVerticalHeaderLabels(self._labels)
            for i in range(n):
                for j in range(n):
                    item = QtWidgets.QTableWidgetItem(f"{self._matrix[i, j]:+.4f}")
                    item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                    self.table.setItem(i, j, item)
            self.table.resizeColumnsToContents()

        def _on_cell_double_clicked(self, item: "QtWidgets.QTableWidgetItem") -> None:
            i = item.row()
            j = item.column()
            if self._matrix is None:
                return
            val = float(self._matrix[i, j])
            self.lbl_status.setText(
                f"({self._labels[i]}, {self._labels[j]}) = {val:+.6f}"
            )

    class EnsembleAnalysisDialog(QtWidgets.QDialog):
        """Dialog para análise estatística do ensemble: mediana + envelope.

        UI:
          • Combo: percentil baixo (1, 5, 10, 25)
          • Combo: percentil alto (75, 90, 95, 99)
          • Combo: componente EM (Hxx..Hzz)
          • Spin TR / Ang / Freq (índices fixados)
          • Plot: mediana (azul) + fill_between(p_low, p_high) cinza
          • Outliers (z>3) em vermelho

        Note:
            Requer ``H_stack`` 6D ``(n_models, nTR, nAng, n_pos, nf, 9)``.
        """

        def __init__(
            self,
            parent: Optional[QtWidgets.QWidget],
            H_stack: np.ndarray,
            *,
            z_obs: Optional[np.ndarray] = None,
            percentiles: Tuple[float, float] = (5.0, 95.0),
        ) -> None:
            super().__init__(parent)
            self.setWindowTitle("Análise — Ensemble: Mediana + Envelope")
            self.setModal(False)
            self.resize(900, 700)
            self._H = np.asarray(H_stack)
            if self._H.ndim != 6:
                raise ValueError(
                    f"EnsembleAnalysisDialog requer H_stack 6D "
                    f"(n_models, nTR, nAng, n_pos, nf, 9), recebeu {self._H.shape}"
                )
            n_models, nTR, nAng, n_pos, nf, _ = self._H.shape
            self._z_obs = z_obs if z_obs is not None else np.arange(n_pos)
            self._envelope: Optional[Dict[str, np.ndarray]] = None

            # Top: percentis + componente
            top = QtWidgets.QHBoxLayout()
            top.addWidget(QtWidgets.QLabel("Percentil baixo:"))
            self.combo_p_low = QtWidgets.QComboBox()
            for p in (1, 5, 10, 25):
                self.combo_p_low.addItem(f"P{p}", p)
            self.combo_p_low.setCurrentText(f"P{int(percentiles[0])}")
            top.addWidget(self.combo_p_low)

            top.addWidget(QtWidgets.QLabel("alto:"))
            self.combo_p_high = QtWidgets.QComboBox()
            for p in (75, 90, 95, 99):
                self.combo_p_high.addItem(f"P{p}", p)
            self.combo_p_high.setCurrentText(f"P{int(percentiles[1])}")
            top.addWidget(self.combo_p_high)

            top.addWidget(QtWidgets.QLabel("Componente:"))
            self.combo_comp = QtWidgets.QComboBox()
            self.combo_comp.addItems(EM_COMPONENT_LABELS)
            self.combo_comp.setCurrentIndex(8)  # Hzz default
            top.addWidget(self.combo_comp)

            self.btn_refresh = QtWidgets.QPushButton("Atualizar")
            self.btn_refresh.clicked.connect(self._on_refresh)
            top.addWidget(self.btn_refresh)
            top.addStretch(1)

            # Spins TR / Ang / Freq
            row2 = QtWidgets.QHBoxLayout()
            self.spin_tr = QtWidgets.QSpinBox()
            self.spin_tr.setRange(0, max(0, nTR - 1))
            self.spin_ang = QtWidgets.QSpinBox()
            self.spin_ang.setRange(0, max(0, nAng - 1))
            self.spin_freq = QtWidgets.QSpinBox()
            self.spin_freq.setRange(0, max(0, nf - 1))
            for label, w in (
                ("Índice TR:", self.spin_tr),
                ("Índice ângulo:", self.spin_ang),
                ("Índice frequência:", self.spin_freq),
            ):
                row2.addWidget(QtWidgets.QLabel(label))
                row2.addWidget(w)
            row2.addStretch(1)

            # Canvas
            try:
                from .sm_plots import EMCanvas, PlotStyle

                self.canvas = EMCanvas(self, figsize=(8, 5), style=PlotStyle())
            except Exception:
                self.canvas = QtWidgets.QLabel("Matplotlib indisponível.")

            self.lbl_status = QtWidgets.QLabel("Pronto.")
            self.lbl_status.setStyleSheet("color:#888888; font-size:11px;")

            root = QtWidgets.QVBoxLayout(self)
            root.addLayout(top)
            root.addLayout(row2)
            root.addWidget(self.canvas, 1)
            root.addWidget(self.lbl_status)
            close_btn = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.StandardButton.Close
            )
            close_btn.rejected.connect(self.reject)
            root.addWidget(close_btn)

            # Computa envelope inicial
            self._on_refresh()

        def _on_refresh(self) -> None:
            try:
                p_low = float(self.combo_p_low.currentData() or 5)
                p_high = float(self.combo_p_high.currentData() or 95)
                self.lbl_status.setText(
                    f"Computando envelope P{p_low:.0f}-P{p_high:.0f}…"
                )
                QtWidgets.QApplication.processEvents()
                self._envelope = compute_ensemble_envelope(
                    self._H, percentiles=(p_low, p_high)
                )
                self._refresh_plot()
                self.lbl_status.setText(
                    f"OK — envelope P{p_low:.0f}-P{p_high:.0f} computado."
                )
            except Exception as exc:
                self.lbl_status.setText(f"Erro: {exc}")

        def _refresh_plot(self) -> None:
            env = self._envelope
            if env is None or not hasattr(self.canvas, "figure"):
                return
            iTR = int(self.spin_tr.value())
            iAng = int(self.spin_ang.value())
            ifq = int(self.spin_freq.value())
            comp_idx = int(self.combo_comp.currentIndex())

            median = env["median"]
            p_low = env["p_low"]
            p_high = env["p_high"]

            # Slice (n_pos,) — usa parte real para plot
            try:
                med_slice = np.real(median[iTR, iAng, :, ifq, comp_idx])
                lo_slice = np.real(p_low[iTR, iAng, :, ifq, comp_idx])
                hi_slice = np.real(p_high[iTR, iAng, :, ifq, comp_idx])
            except Exception as exc:
                self.lbl_status.setText(f"Erro slice: {exc}")
                return

            z = np.asarray(self._z_obs).flatten()[: len(med_slice)]

            fig = self.canvas.figure
            fig.clear()
            ax = fig.add_subplot(111)
            # Profundidade no eixo Y (invertida — z+ desce)
            ax.fill_betweenx(
                z, lo_slice, hi_slice, color="#888888", alpha=0.3, label="Envelope"
            )
            ax.plot(med_slice, z, color="#1f4ea8", lw=1.5, label="Mediana")

            # Outliers (z-score > 3 sobre a média do componente)
            try:
                outliers = detect_outliers_zscore(
                    self._H, threshold=3.0, component_index=comp_idx
                )
                n_out = int(np.sum(outliers))
                if n_out > 0:
                    out_models = self._H[outliers]
                    for m in out_models[: min(5, n_out)]:
                        try:
                            mod_slice = np.real(m[iTR, iAng, :, ifq, comp_idx])
                            ax.plot(mod_slice, z, color="#a3272f", lw=0.8, alpha=0.5)
                        except Exception:
                            continue
                    self.lbl_status.setText(
                        f"OK — envelope + {n_out} modelos outliers (z>3)."
                    )
            except Exception:
                pass

            ax.invert_yaxis()
            ax.set_xlabel(f"{EM_COMPONENT_LABELS[comp_idx]} (Re)")
            ax.set_ylabel("Profundidade z (m)")
            ax.set_title(
                f"Ensemble — {EM_COMPONENT_LABELS[comp_idx]} | "
                f"TR={iTR}, Ang={iAng}, Freq={ifq}"
            )
            ax.legend(loc="best", fontsize=9)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            try:
                self.canvas.canvas.draw_idle()
            except Exception:
                pass

else:  # _QT_OK == False — Qt indisponível, classes vazias para imports não falharem

    class CorrelationAnalysisDialog:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("PyQt6/PyQt5/PySide6 não disponível.")

    class EnsembleAnalysisDialog:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("PyQt6/PyQt5/PySide6 não disponível.")
