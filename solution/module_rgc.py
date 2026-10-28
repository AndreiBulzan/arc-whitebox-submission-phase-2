"""FlopScope inference module for the Shared-Trunk Recurrent Graph-Cumulant (RGC-RNN) corrector.

Streamlined for minimum FlopScope RPC calls and zero fresh (16, n, n) / (14, n, n) allocations
during predict() (F53 residual-time discipline). All numerical operations use flopscope.
"""
from __future__ import annotations

import gc as _gc
from pathlib import Path
import flopscope as fs
import flopscope.numpy as f
from _zero_store import zero32 as _zero32
from module_02 import reserve
from module_05 import mean_gain


def _optimal_t_chunks(bounds, n):
    """Partition layers 0..m into up to 3 contiguous chunks minimizing T-channel FLOPs."""
    m = len(bounds)
    if m == 0:
        return ((0, 0, 0, n),)
    def box(s0, s1):
        r0 = min(bounds[j][0] for j in range(s0, s1))
        r1 = max(bounds[j][1] for j in range(s0, s1))
        nr = max(0, r1 - r0)
        return r0, r1, (s1 - s0) * (32 * nr * r1 + 7 * nr * nr)
    best_cost = box(0, m)[2]
    best_splits = ((0, m),)
    for i in range(1, m):
        c2 = box(0, i)[2] + box(i, m)[2]
        if c2 < best_cost:
            best_cost = c2
            best_splits = ((0, i), (i, m))

    chunks = []
    for a, b in best_splits:
        r0, r1, _ = box(a, b)
        if chunks and chunks[-1][2] == r0 and chunks[-1][3] == r1:
            chunks[-1] = (chunks[-1][0], b, r0, r1)
        else:
            chunks.append((a, b, r0, r1))
    return tuple(chunks)


def _graph_response_split(alpha, sigma, d3, g4, covariance, d21, theta, workspace, bounds=None):
    """Evaluate module_02 graph response returning local-36, 8 separate graph operator channels, and sum."""
    layers, n = alpha.shape
    reserve(workspace, layers, n)
    R, T, M = (workspace[k] for k in ("R", "T", "M"))
    part = workspace["part"]
    op_buf = workspace.get("op_buf")
    if op_buf is None or op_buf.shape != (8, layers, n):
        op_buf = f.empty((8, layers, n), dtype=f.float32)
        workspace["op_buf"] = op_buf
    a = f.clip(alpha.astype(f.float32), -6.0, 6.0)
    s = sigma.astype(f.float32)
    v = s * s
    k3 = f.clip(d3.astype(f.float32) / (v * s), -4.0, 4.0)
    k4 = f.clip(g4.astype(f.float32) / (v * v), -4.0, 4.0)
    phi = f.exp(-0.5 * a * a) * 0.3989422804014327
    a2 = a * a
    a3 = a2 * a
    H = f.stack(
        (f.ones_like(a), a, a2 - 1.0, a3 - 3.0 * a, a2 * a2 - 6.0 * a2 + 3.0, a2 * a3 - 10.0 * a3 + 15.0 * a),
        axis=2,
    )
    powers = f.stack((f.ones_like(a), k3, k4, k3 * k3, k3 * k4, k4 * k4), axis=2)
    sp = s * phi
    theta_loc = workspace.get("theta_loc")
    if theta_loc is None or theta_loc.shape[0] != layers:
        theta_loc = theta[:, :36].reshape(layers, 6, 6)
    val_loc = sp * f.sum((H @ theta_loc) * powers, axis=2)

    nodes = f.stack((f.ones_like(a), a, a2 - 1.0, k3, k4, phi, a * k3), axis=2)
    f.divide(R, s[:, None, :], out=R)
    theta_enc = workspace.get("theta_enc")
    if theta_enc is None or theta_enc.shape[0] != layers:
        theta_enc = f.swapaxes(theta[:, 36:].reshape(layers, 8, 3, 7), 2, 3)
    encoded = nodes[:, None] @ theta_enc
    H3 = H[:, :, :3]
    sp_norm = (sp * 0.03125) if n == 1024 else (sp / f.sqrt(f.asarray(n, dtype=f.float32)))

    f.matmul(R, encoded[:, 0], out=part)
    f.sum(H3 * part, axis=2, out=op_buf[0])

    f.multiply(R, R, out=M)
    f.matmul(M, encoded[:, 3], out=part)
    f.sum(H3 * part, axis=2, out=op_buf[3])

    chunks = _optimal_t_chunks(bounds, n) if (bounds is not None and len(bounds) == layers) else ((0, layers, 0, n),)
    if len(chunks) == 1 and chunks[0][2] == 0 and chunks[0][3] == n:
        transposed = f.swapaxes(T, 1, 2)
        for i, matrix in ((1, T), (2, transposed)):
            f.matmul(matrix, encoded[:, i], out=part)
            f.sum(H3 * part, axis=2, out=op_buf[i])
        for i, left, right in ((4, R, T), (5, None, None), (6, T, T), (7, T, transposed)):
            if i == 5:
                f.matmul(f.swapaxes(M, 1, 2), encoded[:, 5], out=part)
            else:
                f.multiply(left, right, out=M)
                f.matmul(M, encoded[:, i], out=part)
            f.sum(H3 * part, axis=2, out=op_buf[i])
    else:
        f.copyto(op_buf[1:3], _zero32())
        f.copyto(op_buf[4:8], _zero32())
        for s0, s1, r0, r1 in chunks:
            if r0 >= r1:
                continue
            nr = r1 - r0
            T_sub = T[s0:s1, r0:r1, :r1]
            T_sub_T = f.swapaxes(T_sub, 1, 2)
            R_sub = R[s0:s1, r0:r1, :r1]
            M_sub = M[s0:s1, :nr, :r1]
            part_nr = part[s0:s1, :nr, :]
            part_r1 = part[s0:s1, :r1, :]
            H3_nr = H3[s0:s1, r0:r1, :]
            H3_r1 = H3[s0:s1, :r1, :]

            f.matmul(T_sub, encoded[s0:s1, 1, :r1], out=part_nr)
            f.sum(H3_nr * part_nr, axis=2, out=op_buf[1, s0:s1, r0:r1])

            f.matmul(T_sub_T, encoded[s0:s1, 2, r0:r1], out=part_r1)
            f.sum(H3_r1 * part_r1, axis=2, out=op_buf[2, s0:s1, :r1])

            f.multiply(R_sub, T_sub, out=M_sub)
            f.matmul(M_sub, encoded[s0:s1, 4, :r1], out=part_nr)
            f.sum(H3_nr * part_nr, axis=2, out=op_buf[4, s0:s1, r0:r1])

            f.matmul(f.swapaxes(M_sub, 1, 2), encoded[s0:s1, 5, r0:r1], out=part_r1)
            f.sum(H3_r1 * part_r1, axis=2, out=op_buf[5, s0:s1, :r1])

            f.multiply(T_sub, T_sub, out=M_sub)
            f.matmul(M_sub, encoded[s0:s1, 6, :r1], out=part_nr)
            f.sum(H3_nr * part_nr, axis=2, out=op_buf[6, s0:s1, r0:r1])

            T_sq = T[s0:s1, r0:r1, r0:r1]
            M_sq = M[s0:s1, :nr, :nr]
            f.multiply(T_sq, f.swapaxes(T_sq, 1, 2), out=M_sq)
            f.matmul(M_sq, encoded[s0:s1, 7, r0:r1], out=part_nr)
            f.sum(H3_nr * part_nr, axis=2, out=op_buf[7, s0:s1, r0:r1])

    f.multiply(op_buf, sp_norm[None, :, :], out=op_buf)
    val_ops = f.swapaxes(f.swapaxes(op_buf, 0, 1), 1, 2)  # (layers, n, 8)
    val_gr = f.sum(op_buf, axis=0)
    return val_loc, val_gr, val_ops


def _tf(x):
    return f.sign(x) * f.log1p(f.abs(x) * 1e3)


_GELU_C0 = 0.79768205
_GELU_C1 = 0.036738366
_GELU_C2 = -0.00025659447
_GELU_C3 = -8.561639e-06


def _gelu(x, coeffs=None):
    xc = f.clip(x, -6.5, 6.5)
    x2 = xc * xc
    if coeffs is not None:
        p = xc * f.polyval(coeffs, x2)
    else:
        p = xc * (_GELU_C0 + x2 * (_GELU_C1 + x2 * _GELU_C2))
    return x * (1.0 + f.tanh(p))


def install(cls, policy: str = "pooled", weights_filename: str = "rgc_rnn_weights.npz"):
    """Install the Shared-Trunk RGC-RNN corrector on Estimator (replaces module_06.install)."""
    if getattr(cls, "LATE_RESPONSE_ENABLED", False):
        raise RuntimeError("Late response already installed on Estimator.")
    setup0 = cls.setup
    observe0 = cls._higher_observe
    finish0 = cls._finish_response
    cls.LATE_RESPONSE_ENABLED = True
    cls.LATE_RESPONSE_POLICY = policy

    def setup(self, ctx):
        _gc.enable()
        setup0(self, ctx)
        _gc.enable()
        m03 = f.load(str(Path(__file__).with_name("data_03.npz")))
        self._late_theta_layers = m03["layers"].astype(f.float32)
        self._late_workspace = {}
        n_ctx = getattr(ctx, "width", 1024)
        d_ctx = getattr(ctx, "depth", 16)
        if n_ctx == 1024 and d_ctx == 16:
            self._rgc_W_all = f.empty((16, 1024, 1024), dtype=f.float32)
            f.copyto(self._rgc_W_all, _zero32())
            self._rgc_W2_all = f.empty((16, 1024, 1024), dtype=f.float32)
            f.copyto(self._rgc_W2_all, _zero32())
            _t_ws = f.empty((14, 1024, 1024), dtype=f.float32)
            f.copyto(_t_ws, _zero32())
            _part_ws = f.empty((14, 1024, 3), dtype=f.float32)
            f.copyto(_part_ws, _zero32())
            _op_buf_ws = f.empty((8, 14, 1024), dtype=f.float32)
            f.copyto(_op_buf_ws, _zero32())
            self._late_workspace = {
                "geometry": (14, 1024),
                "R": self._rgc_W_all[:14],
                "T": _t_ws,
                "M": self._rgc_W2_all[:14],
                "part": _part_ws,
                "op_buf": _op_buf_ws,
                "theta": f.copy(self._late_theta_layers),
                "theta_loc": f.copy(self._late_theta_layers[:, :36].reshape(14, 6, 6)),
                "theta_enc": f.copy(f.swapaxes(self._late_theta_layers[:, 36:].reshape(14, 8, 3, 7), 2, 3)),
            }
            for _k_ws in ("R", "T"):
                self._late_workspace[f"{_k_ws}_diagonal"] = f.reshape(
                    self._late_workspace[_k_ws], (14, 1024 * 1024)
                )[:, ::1025]
            self._rgc_ones_n = f.ones(1024, dtype=f.float32)
            self._rgc_z_1_n = f.zeros((1, 1024), dtype=f.float32)
            self._rgc_z_1_n_8 = f.zeros((1, 1024, 8), dtype=f.float32)
            self._rgc_z_1_n_24 = f.zeros((1, 1024, 24), dtype=f.float32)
            self._rgc_xv0_buf = f.zeros((1, 1024, 24), dtype=f.float32)
            self._late_t_clean = True
        self._rgc_gelu_coeffs = f.asarray([_GELU_C2, _GELU_C1, _GELU_C0], dtype=f.float32)

        rgc_path = Path(__file__).with_name(weights_filename)
        if rgc_path.exists():
            rgc = f.load(str(rgc_path))
            weights_dict = {}
            for k in (rgc.files if hasattr(rgc, "files") else rgc.keys()):
                if k == "mode":
                    val = rgc[k]
                    raw_m = val.item() if getattr(val, "ndim", 0) == 0 else val[0]
                    if isinstance(raw_m, (int, float)) or hasattr(raw_m, "__index__"):
                        weights_dict[k] = "replace_m06" if int(raw_m) == 1 else "residual_sep_g"
                    else:
                        weights_dict[k] = str(raw_m)
                elif k == "act":
                    val = rgc[k]
                    raw_a = val.item() if getattr(val, "ndim", 0) == 0 else val[0]
                    if isinstance(raw_a, (bytes, bytearray)):
                        raw_a = raw_a.decode("utf-8", errors="ignore")
                    if isinstance(raw_a, (int, float)) or hasattr(raw_a, "__index__"):
                        weights_dict[k] = "relu" if int(raw_a) == 1 else "gelu"
                    else:
                        weights_dict[k] = str(raw_a).lower()
                elif k == "flags":
                    weights_dict[k] = [int(x) for x in rgc[k]]
                else:
                    weights_dict[k] = rgc[k].astype(f.float32)
            W1_full = weights_dict["W1"]
            d_emb_w = int(weights_dict.get("flags", [36, 12, 32, 128, 0, 1])[2])
            weights_dict["W1_60"] = f.copy(W1_full[:60])
            weights_dict["W1_h"] = f.copy(W1_full[60:60 + d_emb_w])
            w1_u = W1_full[60 + d_emb_w:64 + d_emb_w]
            weights_dict["W1_u"] = f.copy(w1_u) if float(f.sum(f.abs(w1_u))) > 0.0 else None
            weights_dict["es_f"] = tuple(float(x) for x in weights_dict["es"])
            weights_dict["inv_es"] = tuple(float(1.0 / x) for x in weights_dict["es"])
            weights_dict["W2"] = 0.5 * weights_dict["W2"]
            weights_dict["W3"] = 0.5 * weights_dict["W3"]
            weights_dict["wo"] = 0.5 * weights_dict["wo"]
            if "wo_15" in weights_dict:
                weights_dict["wo_15"] = 0.5 * weights_dict["wo_15"]
            weights_dict["Wh"] = 0.5 * weights_dict["Wh"]
            weights_dict["b2_2d"] = weights_dict["b2"][None, :]
            weights_dict["b3_2d"] = weights_dict["b3"][None, :]
            weights_dict["bh_2d"] = weights_dict["bh"][None, :]
            weights_dict["mzoL_3d"] = weights_dict["mzoL"][:, None, :]
            weights_dict["izoL_3d"] = weights_dict["izoL"][:, None, :]
            weights_dict["mzvL1_3d"] = weights_dict["mzvL"][1:, None, :]
            weights_dict["izvL1_3d"] = weights_dict["izvL"][1:, None, :]
            weights_dict["b1L_3d"] = weights_dict["b1L"][:, None, :]
            self._rgc_weights = weights_dict
        else:
            self._rgc_weights = None
        controller = self._fold_controller
        birth0 = controller.observe_birth

        def birth(li, d21, g4):
            self._late_d21 = d21
            return birth0(li, d21, g4)

        controller.observe_birth = birth
        _gc.collect()
        _gc.freeze()
        _gc.disable()

    def observe(self, W, mu, a, s, phi, g, d3, g4, cov, sketch, li, depth, inverse, wick):
        result = observe0(self, W, mu, a, s, phi, g, d3, g4, cov, sketch, li, depth, inverse, wick)
        if li == 0:
            self._late_records = []
            self._late_w2_rowsums = []
            self._late_bounds = []
            self._late_start = 1
            self._late_stop = depth - 1
            self._late_enabled = self.LATE_RESPONSE_ENABLED and a.shape[0] >= 32 and depth == 16
            if self._late_enabled:
                count = self._late_stop - self._late_start
                reserve(self._late_workspace, count, a.shape[0])
                if "theta" not in self._late_workspace:
                    self._late_workspace["theta"] = f.copy(self._late_theta_layers)
                z0_buf = getattr(self, "_rgc_z_1_n", None)
                z0 = z0_buf[0] if (z0_buf is not None and z0_buf.shape == (1, mu.shape[0])) else f.zeros_like(mu)
                self._rgc_rec0 = f.take(f.stack((mu, s * s, z0, z0, a, s, phi, g), axis=0), inverse, axis=1)
            return result
        if not self._late_enabled:
            return result
        self._late_records.append(f.stack((mu, s * s, d3, g4, a, s, phi, g), axis=0))
        _w2_rs = getattr(self, "_cur_w2_rowsum", None)
        if _w2_rs is not None:
            self._late_w2_rowsums.append(_w2_rs)
        if li < self._late_stop:
            j = li - self._late_start
            f.divide(cov, s[:, None], out=self._late_workspace["R"][j])
            _r0 = getattr(self, "_query_start", 0)
            _r1 = getattr(self, "_query_end", a.shape[0])
            self._late_bounds.append((int(_r0), int(_r1)))
            _t_j = self._late_workspace["T"][j, _r0:_r1, :_r1]
            f.divide(self._late_d21[_r0:_r1, :_r1], (s[_r0:_r1] * s[_r0:_r1])[:, None], out=_t_j)
            f.divide(_t_j, s[None, :_r1], out=_t_j)
        self._late_d21 = None
        return result

    def finish(self, base, mlp):
        if not self._late_enabled:
            return finish0(self, base, mlp)
        records = f.stack(self._late_records)  # (15, 8, n)
        full_inverse = f.stack(self._higher_inverses, axis=0)  # (15, n)
        self._higher_inverse_stack = full_inverse
        count = self._late_stop - self._late_start  # 14
        ws = self._late_workspace
        _r_bounds = getattr(self, "_late_bounds", None)
        if _r_bounds:
            _T_ws = ws["T"]
            for _g0, _g1, _gr0, _gr1 in _optimal_t_chunks(_r_bounds, mlp.width):
                for _j in range(_g0, _g1):
                    _jr0, _jr1 = _r_bounds[_j]
                    if _jr0 > _gr0:
                        f.copyto(_T_ws[_j, _gr0:_jr0, :_gr1], _zero32())
                    if _jr1 < _gr1:
                        f.copyto(_T_ws[_j, _jr1:_gr1, :_gr1], _zero32())
                        f.copyto(_T_ws[_j, _jr0:_jr1, _jr1:_gr1], _zero32())
        val_loc, val_gr, val_ops = _graph_response_split(
            records[:count, 4],
            records[:count, 5],
            records[:count, 2],
            records[:count, 3],
            ws["R"],
            ws["T"],
            ws["theta"],
            ws,
            _r_bounds,
        )
        forcing_loc = f.take_along_axis(val_loc, full_inverse[:count], axis=1)
        forcing_gr = f.take_along_axis(val_gr, full_inverse[:count], axis=1)
        forcing_ops = f.take_along_axis(val_ops, full_inverse[:count, :, None], axis=1)
        forcing = forcing_loc + forcing_gr

        marginals = records[:, :4].astype(f.float64)
        gains = mean_gain(marginals[:, 0], marginals[:, 1], marginals[:, 2], marginals[:, 3])
        gains = f.take_along_axis(gains, full_inverse, axis=1)

        output_m7 = finish0(self, base, mlp)
        m7_errors = getattr(self, "_m7_errors", None)
        m7_recal_1_15 = m7_errors if m7_errors is not None else (output_m7 - base)[1:]
        f.maximum(output_m7, 0.0, out=output_m7)
        base_after_m7_15 = f.copy(output_m7[15])

        delta = f.zeros(mlp.width, dtype=f.float64)
        wt = self._readout_weight_transposes
        deltas = []
        w_deltas_32 = []
        for li in range(1, mlp.depth):
            if li > 1:
                wd_cur = wt[li] @ delta
                w_deltas_32.append(wd_cur.astype(f.float32))
                delta = gains[li - 1] * wd_cur
            if li < self._late_stop:
                delta = delta + forcing[li - 1]
            deltas.append(delta)
        deltas_st = f.stack(deltas, axis=0).astype(f.float32)
        f.copyto(output_m7[1:], (output_m7[1:].astype(f.float64) + deltas_st.astype(f.float64)).astype(f.float32))

        if self._rgc_weights is not None and mlp.depth == 16 and mlp.width == 1024:
            rw = self._rgc_weights
            n = mlp.width
            rec_nat = f.take_along_axis(records, full_inverse[:, None, :], axis=2)
            all_rec = f.concatenate((self._rgc_rec0[None], rec_nat), axis=0)

            flags = rw.get("flags", [36, 12, 32, 96, 0, 0])
            fo, fp, d_emb, _d_hid, xall_flag, embw_flag = (int(x) for x in flags[:6])

            mzoL_3d, izoL_3d = rw["mzoL_3d"], rw["izoL_3d"]
            mzvL1_3d, izvL1_3d = rw["mzvL1_3d"], rw["izvL1_3d"]
            es = rw["es_f"]
            inv_es = rw["inv_es"]
            W1_60, W1_h, b1L_3d = rw["W1_60"], rw["W1_h"], rw["b1L_3d"]
            W1_u = rw.get("W1_u")
            W1_h_A = rw.get("W1_h_A")
            W1_h_B = rw.get("W1_h_B")
            W2_mlp, b2_2d = rw["W2"], rw["b2_2d"]
            W3_mlp, b3_2d = rw["W3"], rw["b3_2d"]
            wo, bo = rw["wo"], rw["bo"]
            wo_15 = rw.get("wo_15", wo)
            bo_15 = rw.get("bo_15", bo)
            Wh, bh_2d = rw["Wh"], rw["bh_2d"]
            mode_str = rw.get("mode", "residual_sep_g")

            g_edge_1_15 = gains.astype(f.float32)

            W_all = getattr(self, "_rgc_W_all", None)
            if W_all is None or W_all.shape != (16, n, n):
                W_all = f.stack([mlp.weights[0].T, *wt[1:16]], axis=0)
                W2_all = W_all * W_all
            else:
                f.stack([mlp.weights[0].T, *wt[1:16]], axis=0, out=W_all)
                W2_all = f.multiply(W_all, W_all, out=self._rgc_W2_all)

            s1_list = [None] * 16
            s2_list = [None] * 16
            w1_list = [None] * 16
            s1_cur = getattr(self, "_rgc_ones_n", f.ones(n, dtype=f.float32))
            s2_cur = s1_cur
            w1_cur = s1_cur
            s1_list[15] = s1_cur
            s2_list[15] = s2_cur
            w1_list[15] = w1_cur
            for lb in range(14, -1, -1):
                W_next = wt[lb + 1]
                W2_next = W2_all[lb + 1]
                g_next = g_edge_1_15[lb]
                s1_cur = W_next.T @ (g_next * s1_cur)
                rhs_b2 = f.stack(((g_next * g_next) * s2_cur, w1_cur), axis=1)
                out_b2 = W2_next.T @ rhs_b2
                s2_cur = out_b2[:, 0]
                w1_cur = out_b2[:, 1]
                s1_list[lb] = s1_cur
                s2_list[lb] = s2_cur
                w1_list[lb] = w1_cur

            mu_all = all_rec[:, 0]
            v_all = all_rec[:, 1]
            d3_all = all_rec[:, 2]
            g4_all = all_rec[:, 3]
            a_all = f.clip(all_rec[:, 4], -6.0, 6.0)
            sg_all = all_rec[:, 5]
            phi_all = all_rec[:, 6]
            Phi_all = all_rec[:, 7]
            g_ed_all = f.concatenate((Phi_all[:1], g_edge_1_15), axis=0)

            z_1_n = getattr(self, "_rgc_z_1_n", f.zeros((1, n), dtype=f.float32))
            z_1_n_8 = getattr(self, "_rgc_z_1_n_8", f.zeros((1, n, 8), dtype=f.float32))
            z_1_n_24 = getattr(self, "_rgc_z_1_n_24", f.zeros((1, n, 24), dtype=f.float32))

            k3_all = f.concatenate((z_1_n, f.clip(d3_all[1:] / (v_all[1:] * sg_all[1:]), -4.0, 4.0)), axis=0)
            k4_all = f.concatenate((z_1_n, f.clip(g4_all[1:] / (v_all[1:] * v_all[1:]), -4.0, 4.0)), axis=0)

            a2_all = a_all * a_all
            a3_all = a2_all * a_all
            h2_all = a2_all - 1.0
            h3_all = a3_all - 3.0 * a_all
            h4_all = f.clip(a2_all * a2_all - 6.0 * a2_all + 3.0, -20.0, 20.0)
            h5_all = f.clip(a3_all * a2_all - 10.0 * a3_all + 15.0 * a_all, -60.0, 60.0)

            if len(getattr(self, "_late_w2_rowsums", ())) == 15:
                w2_rs_1_15 = f.take_along_axis(f.stack(self._late_w2_rowsums, axis=0), full_inverse, axis=1)
                w_rownorm_all = f.concatenate((sg_all[:1], f.sqrt(w2_rs_1_15)), axis=0) - 1.0
            else:
                w_rownorm_all = f.sqrt(f.sum(W2_all, axis=2)) - 1.0
            w_rowsum_all = f.sum(W_all, axis=2)

            w_m7_cached = getattr(self, "_m7_we_1_14", None)
            w_m7_1_14 = w_m7_cached if w_m7_cached is not None else (W_all[2:16] @ m7_recal_1_15[:14, :, None])[:, :, 0]
            m7_fr_2_15 = m7_recal_1_15[1:15] - Phi_all[2:16] * w_m7_1_14
            m7_fr_all = f.concatenate((z_1_n, m7_recal_1_15[:1], m7_fr_2_15), axis=0)
            m7_cm_all = f.concatenate((z_1_n, m7_recal_1_15), axis=0)
            fl_loc_all = f.concatenate((z_1_n, forcing_loc, z_1_n), axis=0)
            fl_gr_all = f.concatenate((z_1_n, forcing_gr, z_1_n), axis=0)
            fl_ops_all = f.concatenate((z_1_n_8, forcing_ops, z_1_n_8), axis=0)
            dl_e13_all = f.concatenate((z_1_n, deltas_st), axis=0)
            s1_all = f.stack(s1_list, axis=0)
            s2_all = f.stack(s2_list, axis=0)
            w1_m1_all = f.stack(w1_list, axis=0) - 1.0

            X0_all = f.stack(
                (
                    mu_all, v_all, sg_all, a_all, phi_all, Phi_all, g_ed_all - Phi_all, base,
                    d3_all, g4_all, k3_all, k4_all, h2_all, h3_all, h4_all, h5_all,
                    w_rownorm_all, w_rowsum_all, phi_all * a_all, phi_all * k3_all, phi_all * k4_all,
                    m7_fr_all, m7_cm_all, fl_loc_all,
                    fl_ops_all[:, :, 0], fl_ops_all[:, :, 1], fl_ops_all[:, :, 2], fl_ops_all[:, :, 3],
                    fl_ops_all[:, :, 4], fl_ops_all[:, :, 5], fl_ops_all[:, :, 6], fl_ops_all[:, :, 7],
                    dl_e13_all, s1_all, s2_all, w1_m1_all,
                ),
                axis=2,
            )
            Z0_all = (_tf(X0_all) - mzoL_3d) * izoL_3d

            # At l=0, mu_all[0]==0, a_all[0]==0, and base[0] == 0.3989422804014327 * sg_all[0]
            xp0_2 = f.stack((v_all[0], sg_all[0]), axis=1)
            xv0_w1 = W_all[1] @ xp0_2
            xv0_w2 = W2_all[1] @ xp0_2
            xv0_buf = getattr(self, "_rgc_xv0_buf", f.zeros((1, n, 24), dtype=f.float32))
            f.copyto(xv0_buf[0, :, 1:3], xv0_w1)
            f.copyto(xv0_buf[0, :, 4], xv0_w1[:, 1] * 0.3989422804014327)
            f.copyto(xv0_buf[0, :, 13:15], xv0_w2)
            f.copyto(xv0_buf[0, :, 16], xv0_w2[:, 1] * 0.3989422804014327)

            w_dl_1_14 = f.stack(w_deltas_32, axis=0)
            XP_1_14_9 = f.stack(
                (
                    mu_all[1:15], v_all[1:15], sg_all[1:15], a_all[1:15],
                    d3_all[1:15], g4_all[1:15], m7_fr_all[1:15],
                    fl_loc_all[1:15], fl_gr_all[1:15],
                ),
                axis=2,
            )
            xv1_w1_9 = W_all[2:16] @ XP_1_14_9
            xv1_w1 = f.concatenate(
                (
                    xv1_w1_9[:, :, :4],
                    mu_all[2:16, :, None],
                    xv1_w1_9[:, :, 4:7],
                    w_m7_1_14[:, :, None],
                    xv1_w1_9[:, :, 7:9],
                    w_dl_1_14[:, :, None],
                ),
                axis=2,
            )
            XP_1_14_12 = f.concatenate(
                (
                    XP_1_14_9[:, :, :4],
                    base[1:15, :, None],
                    XP_1_14_9[:, :, 4:7],
                    m7_cm_all[1:15, :, None],
                    XP_1_14_9[:, :, 7:9],
                    dl_e13_all[1:15, :, None],
                ),
                axis=2,
            )
            xv1_w2 = W2_all[2:16] @ XP_1_14_12
            xv_raw_all = f.concatenate((xv0_buf, f.concatenate((xv1_w1, xv1_w2), axis=2)), axis=0)
            zv_1_15 = (_tf(xv_raw_all) - mzvL1_3d) * izvL1_3d
            zv_all = f.concatenate((z_1_n_24, zv_1_15), axis=0)
            base_hid_all = f.concatenate((Z0_all, zv_all), axis=2) @ W1_60 + b1L_3d

            h_prev = None
            u_prev = None
            r_trans = int(W1_h_A.shape[1]) if W1_h_A is not None else d_emb

            for l in range(16):
                if l == 0:
                    hid_in = base_hid_all[0]
                    wu = None
                else:
                    if W1_h_A is not None:
                        h_proj = h_prev @ W1_h_A
                        out_hu = W_all[l] @ f.concatenate((h_proj, u_prev[:, None]), axis=1)
                        h_trans_r = out_hu[:, :r_trans]
                        wu = out_hu[:, r_trans]
                        hid_in = base_hid_all[l] + h_trans_r @ W1_h_B
                    else:
                        out_hu = W_all[l] @ f.concatenate((h_prev, u_prev[:, None]), axis=1)
                        h_trans = out_hu[:, :d_emb]
                        wu = out_hu[:, d_emb]
                        hid_in = base_hid_all[l] + h_trans @ W1_h
                    if W1_u is not None:
                        u_scaled = u_prev * inv_es[l - 1]
                        w2u2 = W2_all[l] @ (u_scaled * u_scaled)
                        wu_s = wu * inv_es[l]
                        u_feat = f.stack((wu_s, Phi_all[l] * wu_s, g_ed_all[l] * wu_s, w2u2), axis=1)
                        hid_in = hid_in + u_feat @ W1_u

                g_coeffs = getattr(self, "_rgc_gelu_coeffs", None)
                hid = _gelu(hid_in, g_coeffs)
                hid = _gelu(hid @ W2_mlp + b2_2d, g_coeffs)
                hid = _gelu(hid @ W3_mlp + b3_2d, g_coeffs)
                if l < 15:
                    g_out = (hid @ wo + bo) * es[l]
                    u_cur = g_out if l == 0 else (g_ed_all[l] * wu + g_out)
                    h_prev = f.tanh(hid @ Wh + bh_2d)
                    u_prev = u_cur
                else:
                    g_out = (hid @ wo_15 + bo_15) * es[l]
                    u_cur = g_ed_all[l] * wu + g_out
                    base_15 = base_after_m7_15 if mode_str == "replace_m06" else output_m7[15]
                    f.copyto(output_m7[15], f.maximum(base_15 + u_cur, 0.0))

        self._late_records = []
        self._late_w2_rowsums = []
        self._late_bounds = []
        self._late_d21 = None
        self._rgc_rec0 = None
        self._m7_we_1_14 = None
        self._m7_errors = None
        return output_m7

    cls.setup = setup
    cls._higher_observe = observe
    cls._finish_response = finish


install_rgc = install
