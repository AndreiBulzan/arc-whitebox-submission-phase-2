"""Raw-accuracy research model with implicit mixed fourth states.

Tracked baseline SHA256 f39601edb3c54942d60404a6b8828436602b73bad7d414cc87d2b872425e3dd3.
Independent closure SHA256 b9a035568dec287e73598ad1e627f78c9ba3a82006d281215fbfea3b0a2f6a2b.
All numerical prediction operations use FlopScope; no benchmark labels.
"""
from __future__ import annotations
import math
import flopscope as flops

def _readout_features(cp, pair, alpha, sigma, phi, gain, skew, kurt, include_higher=False, pair_sums=None, return_context=False):
    """Batch independent head columns, retaining each scalar operation order."""
    v = (sigma * sigma).astype(fnp.float64)
    converted = fnp.stack((alpha, sigma, phi, skew, kurt, gain)).astype(fnp.float64)
    a, s, density, k3, k4, g = (converted[index] for index in range(6))
    n = alpha.shape[0]
    hs = [fnp.ones_like(a), a]
    for order in range(1, 8):
        hs.append(a * hs[-1] - order * hs[-2])
    sdensity = s * density
    scale = sdensity / n
    row2 = fnp.sum(cp * cp, axis=1).astype(fnp.float64)
    geometric = row2 / (n * v * v)
    geometric = geometric - fnp.mean(geometric)
    if pair_sums is None:
        pair_row = fnp.sum(pair, axis=1).astype(fnp.float64)
        pair_col = fnp.sum(pair, axis=0).astype(fnp.float64)
    else:
        pair_row, pair_col = (pair_sums[0].astype(fnp.float64), pair_sums[1].astype(fnp.float64))
    # Integer powers use the original scalar-exponent kernels. Repeated powers
    # are shared within this call; no prediction-dependent cache is introduced.
    s5, s6, s7 = (s ** 5, s ** 6, s ** 7)
    polys = [-a, a * a - 1, hs[4], -hs[5], hs[6]]
    denominators = [v, v * s, s5, s6, s7]
    first = [k3, k4, k3, k3, k4]
    second = [k3, k4, k4]
    factors = [6, 24, 72, 144, 1152]
    if include_higher:
        polys += [-hs[3], hs[4]]
        denominators += [s6, s7]
        first += [k3, k4]
        second += [k4, k4]
        factors += [120, 720]
    numerators = fnp.stack(polys) * density[None, :]
    normalized = numerators / fnp.stack(denominators)
    normalized = normalized * fnp.stack(first)
    fnp.multiply(normalized[2:], fnp.stack(second), out=normalized[2:])
    normalized = normalized / fnp.asarray(factors, dtype=fnp.float64)[:, None]
    harmonics = fnp.stack(hs[1:9]) * scale[None, :]
    geoscale = sdensity * geometric
    pair_columns = (fnp.stack((pair_row, pair_col)) * density[None, :]
                    / (n * v)[None, :])
    values = [scale, normalized[0], normalized[1]] + [harmonics[index] for index in range(8)]
    values += [g / n, normalized[2], normalized[3], normalized[4],
               geoscale, geoscale * hs[2], pair_columns[0], pair_columns[1]]
    if include_higher:
        values += [normalized[5], normalized[6]]
    features = fnp.stack(values, axis=1)
    if return_context:
        return features, (a, s, density, numerators[:2])
    return features

def _age_readout_features(base, alpha, sigma, phi, third_cup, third_path, third_repair, fourth_cup, fourth_star, fourth_path, context=None):
    """Batch seven independent age products; preserve every bucket's sum order."""
    if context is None:
        a, s, density = (alpha.astype(fnp.float64), sigma.astype(fnp.float64), phi.astype(fnp.float64))
        h3 = -a * density / (s * s) / 6
        h4 = (a * a - 1) * density / (s * s * s) / 24
    else:
        a, s, density, numerators = context
        square = s * s
        normalizers = numerators / fnp.stack((square, square * s))
        normalizers = normalizers / fnp.asarray([6, 24], dtype=fnp.float64)[:, None]
        h3, h4 = (normalizers[0], normalizers[1])
    recent = third_path[::-1]
    zero = fnp.zeros_like(a)
    age_values = [third_cup, fourth_cup]
    for lo, hi in [(0, 1), (1, 2), (2, 4), (4, 8), (8, 16)]:
        value = zero
        for part in recent[lo:hi]:
            value = value + part
        age_values.append(value)
    products = fnp.stack(age_values) * fnp.stack((h3, h4, h3, h3, h3, h3, h3))
    return fnp.concatenate([base[:, :19], products.T, base[:, 19:]], axis=1)
READOUT_COEFFICIENTS = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.011414201826882815, 0.024009997081421822, -0.3743286849876581, 0.08320640796354857, 0.10269573743686394, 0.02535518150579354, 0.0027348382759705184, -0.00021366414431382466, -0.003471537564583002, -0.0001926508102882803, -0.00020322023262200848, -0.030349259868312342, -0.9347301714153979, -2.8619758180424917, -74.36352539073903, 0.00741129511417053, 0.017294108827488264, -0.0076408401472301485, 0.00043967494709509065, 0.35290210257369803, -0.3743286509590193, 0.013119508031693199, 0.0, 0.0, 0.0, 0.0, 1.3991748352596727, -812.633096050388], [0.06775545199930615, 0.002728020034479598, 0.039133989961144824, 0.07047270992671655, 0.024009934683427353, 0.035283246686079774, -0.008676762474225169, 0.003000504993246624, 0.00014799321419445982, -9.784640980355246e-05, 0.0001694994451952211, -0.01619123777757639, -1.0913384999425295, -6.226181805557635, 29.567175196302646, -0.004219980185376905, -0.016150995932637788, -0.00044557001298954, -5.599044051853568e-05, -0.026045310488015568, 0.039133880134501434, 0.0004848479742541335, 0.011984821078949015, 0.0, 0.0, 0.0, -35.995330053428134, 150.85722814903673], [0.06652125323136147, -0.0008373664122215684, 0.04608329832039768, 0.1036655006754182, -0.021157634650514876, 0.07997369649413873, -0.03114007168833638, 0.016665237232222907, 0.005509225970722892, 0.000991182626889607, 0.0005251548687083659, -0.014716289981754352, -1.2334212038315688, -5.787354330073906, 7.118846142926792, 0.0018664989025185712, -0.007801641741514361, 0.01344269608451202, -0.0006069346578628243, -0.013923393806810154, 0.04608386686688914, -0.0056953420075344895, 0.00206210598565527, 0.015036557554178862, 0.0, 0.0, -32.17170778730157, 145.84918816960464], [0.03214905049871738, -0.004305108535869664, 0.05967651298171428, 0.08812917293990792, -0.1296006243389764, 0.07479167933595868, -0.07107822363602269, 0.015566965190434793, 0.0024928473658989677, 0.0008551702877487342, 0.0007766404279467117, -0.011880986412633748, -0.6625120994258612, -5.006927431576478, 15.203034233918268, 0.0052327400582209424, -0.01352931620394914, -0.011510292482029493, 0.0008446600490799922, -0.030125034747609268, 0.0596759857388184, -0.008409060157545405, -0.002661980224759123, 0.005690870260453531, 0.0, 0.0, -27.706313857071024, 131.86628984940333], [0.010371566929987702, -0.0043857790453332435, 0.009768811541236058, 0.06882664076832463, -0.12483498956689003, 0.06387166514917991, -0.10216294346767588, 0.014891001335331455, -0.002806382965084124, 0.0009988168562736247, 0.000421602597518276, -0.007701559573846291, -1.0961921977680276, -4.9675319770538415, 9.310531239503119, -0.001281743400649488, -0.02165732630176904, -0.004240470873871133, 0.0001342411641515235, -0.023086159768812408, 0.009770038991898317, -0.01238414902744223, -0.006876189235886439, 0.008966093722808165, 0.017913380052925892, 0.0, -27.618084196019566, 82.57353937309757], [-0.023059269223724792, -0.005056937211271927, -0.04390021075089624, 0.037372921664007776, -0.09735389777986442, 0.035462404943143615, -0.10187689634334048, 0.0067846966546985564, -0.0039590698684318975, 0.00031081102745737495, 0.00019768945372124707, -0.004168035632163598, -0.9868931457665117, -4.601320303748375, -0.18071348163856424, -0.0006622557075875223, -0.014629425250976501, 0.010183918762693554, 0.0005213090227223227, -0.023001058559398154, -0.043899912085528466, -0.014139118393936528, -0.0073486405400399754, 0.004931487608032701, 0.013160275337158056, 0.0, -26.081475122936453, 21.023790830887144], [-0.02162828742723095, -0.004702041313865089, 0.035545908994530535, 0.04284361276256573, -0.2825916307227778, 0.04180850630766205, -0.13793370806796887, 0.011251658510555478, -0.004568624587426908, 0.0009055307210005272, 0.0007167336306480749, -0.004139745835456553, -0.8588175738795734, -4.279953447286522, 6.180774192582033, -0.0012752402196399585, -0.01544650375838277, -0.0019232130729451083, -5.7789211297826036e-05, -0.014771069959177058, 0.03554572406556587, -0.017027489888185296, -0.00901309712521904, 0.004867283454617703, 0.008588941664893265, 0.0, -24.02560726737942, 51.234825820896184], [-0.03622022225636951, -0.005003790596229057, 0.015895436200456792, 0.04468333919936588, -0.30455748705156527, 0.04818195636038145, -0.14815583222847173, 0.011971795663690023, -0.007529833236958815, 0.0007736967693513099, 0.0002509141404711709, -0.0038052993092272855, -0.9766159655233018, -4.769895169400778, -0.20423170905400456, 0.005758283486699421, -0.012401504354342906, 0.009167507515469436, 0.0007434143753748069, -0.008722407482476163, 0.015894824828668937, -0.017941220293819612, -0.006477911877838418, -0.009134263237800148, 0.010576748742704643, 0.0, -25.277623910867824, 18.30982398213131], [-0.01524500257824017, -0.0035833533324740392, 0.004224987531168417, 0.046693051665326765, -0.24491862536462522, 0.05925934520205928, -0.1272806893269556, 0.0195887926451786, -0.004715424301987222, 0.0018980452181874335, 0.0007134256242122872, -0.003290914947454355, -1.0757986905658323, -4.513905977620334, 1.7852662853772068, -0.007694338399072551, -0.02005718521235574, 0.002855877918787595, 0.00027243996863772203, 0.006743391252862972, 0.004224397252286962, -0.02157863575656973, -0.015447864893667741, -0.0071145685490885575, 0.009448029492960792, 0.04491717908273572, -24.72083217699602, 11.761252228987523], [-0.046128063539886535, -0.004023554508188537, 0.11906731375722023, 0.011463261887917034, -0.7584150026634974, -0.008954691581183413, -0.20491465747553403, -0.007445973409485715, -0.01900632101149496, -0.0010591608751273289, 0.00034313390925583265, -0.0016108139613346482, -1.387660388723961, -5.425163812039911, 4.553799354799317, 0.0034147518361589742, -0.030889537087675358, 0.002196706659476432, -0.00020578530091625073, 0.0007899074947529575, 0.11906618628606758, -0.016995795163022034, -0.011510478595343248, -0.009646692212100532, 0.0037131095412820518, 0.025414001332803367, -28.21487944340009, 16.661540140072653], [-0.024961142352248205, -0.0036526456843583126, 0.12985445026385295, 0.02128069138204229, -0.8472689461401737, 0.03783050503999454, -0.21601223191914373, 0.012522942029019194, -0.02305514309175404, 0.0009004753401192317, 0.00027636139436980285, -0.001857873648794499, -1.2574551990989442, -5.488968840119247, 4.527627333002207, -0.0033598038944157383, -0.03256854842284677, -0.02302683769335267, 0.00037008022545420533, 0.02036931572809612, 0.1298545067587547, -0.020423298694694014, -0.01833769844263408, -0.01425966079844699, -0.00870604634412181, 0.020989829629670668, -29.22270021951727, 0.6473565898289135], [-0.005726776728800606, -0.002327641968118174, 0.19704495135021502, 0.040377145302424734, -0.9944682923470543, 0.055286778637802854, -0.2345745292636221, 0.015875071139086917, -0.017851644816846805, 0.0011757470937196813, 0.0005928358554123771, -0.0028789703626097463, -1.0388326946656274, -4.610917768784566, 6.4518719881349975, 0.0033741981753372197, -0.012190501765421408, 0.00032489950119823577, 0.002388890518822576, 0.023280229731583803, 0.19704519968072973, -0.021618160903349583, -0.021658387974333327, -0.014171653446055681, 0.0012693764242440691, 0.022701611958757247, -25.793386197116536, 29.098237113227167], [-0.049779664644468584, -0.004220901408971442, 0.21626250906249914, 0.0476080938463327, -1.160968865221061, 0.04549253493384077, -0.2830239123112947, 0.012007973087888025, -0.027699319913710878, 0.0012835984119101444, 0.0012406335850005865, -0.0022796464612107456, -0.9828668658826752, -4.706521309041868, 13.542713122136204, 0.007293560892614746, -0.026532982294499016, -0.02100807443460788, 9.067582067882275e-05, 0.015570800419197018, 0.21626362901995597, -0.016218340246670595, -0.01995070319438411, -0.023201535798398298, -0.003873519959840563, 0.02186356870415435, -26.143182265303544, 43.211357577978745], [-0.020706955606032928, -0.0005724409065532156, 0.26111171134732647, 0.03834864895217591, -1.2969085008182066, 0.014424423483698726, -0.26534520854989774, 0.0027134761143571356, -0.020825315301663056, 0.000222541147228561, 0.00035282211986504495, -0.0015464209087441207, -1.1181728900412917, -4.304846679185241, 5.523458208931628, 0.013670647918179605, -0.01401152164169253, 0.003133105623241564, 0.0008873190442910903, 0.01863161165210997, 0.2611109963851186, -0.014171602437745269, -0.02496857166667856, -0.009292368013815201, 0.0008663125612839845, 0.034578371944023426, -24.905149321293347, 32.796553095504606], [0.005545345757316475, -0.0004578953818777281, 0.42422533550980207, 0.05947816467958691, -1.924035402083075, 0.0750251668307138, -0.29138669340108386, 0.02439664349726425, -0.02673482701536405, 0.002414488722218258, 0.0007255373681779043, -0.0024466961501982414, -1.0613719494298852, -4.83870907957019, 8.52968466222734, 0.006097477506726597, -0.025510006462990877, -0.0024271340196088493, 0.0002980507863399106, 0.04686216972757747, 0.4242248460382017, -0.02329178679293221, -0.027169630098860028, -0.02034758079505218, -0.006738416890195311, 0.011698406092666874, -26.619561926413233, 34.16917184039458]]
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

def _cdf(value):
    return flops.stats.norm.cdf(value).astype(fnp.float32)
_derivative_cache = {}
_covariance_cache = {}

def derivative(p, k, sigma, phi, prob, hs, rs):
    key = (p, k)
    if key not in _derivative_cache:
        _derivative_cache[key] = _derivative_uncached(p, k, sigma, phi, prob, hs, rs)
    return _derivative_cache[key]

def covariance_hermite(cov, sigma, phi, prob, hs, degree):
    if degree not in _covariance_cache:
        _covariance_cache[degree] = _covariance_hermite_uncached(cov, sigma, phi, prob, hs, degree)
    return _covariance_cache[degree].copy()
PREDICTION_CONFIG = {'coherent_pair_strength': 0, 'fourth_path_rank': 0, 'direct_horizon': 0, 'horizon': 16, 'fourth_factor_horizon': 0, 'third_matrix_banks': 16}

def hermites(a, upto):
    hs = [fnp.ones_like(a), a]
    for k in range(1, upto):
        hs.append(a * hs[-1] - k * hs[-2])
    return hs[:upto + 1]

def raw_moments(mu, var, sigma, phi, prob, upto=4):
    rs = [prob, mu * prob + sigma * phi]
    for k in range(2, upto + 1):
        rs.append(mu * rs[-1] + (k - 1) * var * rs[-2])
    return rs

def _derivative_uncached(p, k, sigma, phi, prob, hs, rs):
    if k < p:
        return math.factorial(p) / math.factorial(p - k) * rs[p - k]
    if k == p:
        return math.factorial(p) * prob
    return math.factorial(p) * (-1) ** (k - p - 1) * hs[k - p - 1] * phi / sigma ** (k - p)

def _covariance_hermite_uncached(cov, sigma, phi, prob, hs, degree):
    rho = cov / (sigma[:, None] * sigma[None, :])
    rho = fnp.clip(rho, -1.0, 1.0)
    power = rho.copy()
    result = power * (prob[:, None] * prob[None, :])
    for k in range(2, degree + 1):
        fnp.multiply(power, rho, out=power)
        coeff = phi * hs[k - 2] / math.sqrt(math.factorial(k))
        fnp.add(result, power * (coeff[:, None] * coeff[None, :]), out=result)
    return result * (sigma[:, None] * sigma[None, :])

def _review_gaussian_third_factors_original(cp, sigma, phi, prob, hs, rs, degree=4):
    """Two symmetric CP families for leading distinct terms + exact pair slices."""
    n = len(sigma)
    mean = rs[1]
    k3 = rs[3] - 3 * mean * rs[2] + 2 * mean ** 3
    rho = cp / (sigma[:, None] * sigma[None, :])
    fnp.fill_diagonal(rho, 0.0)
    pair = fnp.zeros_like(cp)
    power = fnp.ones_like(cp)
    for k in range(1, degree + 1):
        power = power * rho
        f_k = derivative(1, k, sigma, phi, prob, hs, rs)
        g_k = derivative(2, k, sigma, phi, prob, hs, rs) - 2 * mean * f_k
        pair = pair + (g_k * sigma ** k)[:, None] * (f_k * sigma ** k)[None, :] * power / math.factorial(k)
    h = phi / sigma
    b = prob[:, None] * cp
    a = fnp.diag(3 * h)
    path21 = 2 * h[:, None] * prob[:, None] * prob[None, :] * fnp.diag(cp)[:, None] * cp
    path21 = path21 + h[None, :] * prob[:, None] ** 2 * cp ** 2
    adjust = pair - path21
    fnp.fill_diagonal(adjust, (k3 - 3 * h * prob ** 2 * fnp.diag(cp) ** 2) / 3)
    return (a, b, pair)

def fourth_birth_trace_matrix(cp, sigma, phi, gain, hs, rs, degree=8, offdiag=True, covariance_degree=None):
    """Trace over final two indices of Gaussian-generated fourth cumulants."""
    n = len(sigma)
    mean = rs[1]
    variance = rs[2] - mean * mean
    central4 = rs[4] - 4 * mean * rs[3] + 6 * mean ** 2 * rs[2] - 3 * mean ** 4
    k4diag = central4 - 3 * variance * variance
    post_cov = covariance_hermite(cp, sigma, phi, gain, hs, degree if covariance_degree is None else covariance_degree)
    fnp.fill_diagonal(post_cov, variance)
    rho = fnp.clip(cp / (sigma[:, None] * sigma[None, :]), -1, 1)
    fnp.fill_diagonal(rho, 0)
    power = fnp.ones_like(cp)
    square_cov = fnp.zeros_like(cp)
    cube_cov = fnp.zeros_like(cp)
    for order in range(1, degree + 1):
        power = power * rho
        f = derivative(1, order, sigma, phi, gain, hs, rs)
        square = derivative(2, order, sigma, phi, gain, hs, rs) - 2 * mean * f
        cube = derivative(3, order, sigma, phi, gain, hs, rs) - 3 * mean * derivative(2, order, sigma, phi, gain, hs, rs) + 3 * mean ** 2 * f
        sf = f * sigma ** order
        ss = square * sigma ** order
        sc = cube * sigma ** order
        square_cov = square_cov + power * ss[:, None] * ss[None, :] / math.factorial(order)
        cube_cov = cube_cov + power * sc[:, None] * sf[None, :] / math.factorial(order)
    k22 = square_cov - 2 * post_cov ** 2
    fnp.fill_diagonal(k22, k4diag)
    trace_diagonal = fnp.sum(k22, axis=1, dtype=fnp.float64).astype(fnp.float32)
    if not offdiag:
        return fnp.diag(trace_diagonal)
    k31 = cube_cov - 3 * variance[:, None] * post_cov
    fnp.fill_diagonal(k31, k4diag)
    h = phi / sigma
    q = -hs[1] * phi / sigma ** 2
    v = fnp.diag(cp)
    d2 = gain * gain
    squared = cp * cp
    row_squares = fnp.sum(squared * d2[None, :], axis=1)
    u = q * row_squares
    result = cp * (u[:, None] * gain[None, :] + gain[:, None] * u[None, :])
    result = result + 2 * gain[:, None] * gain[None, :] * (cp * (q * gain * v)[None, :] @ cp)
    result = result + 2 * h[:, None] * h[None, :] * cp * (cp * d2[None, :] @ cp)
    r = cp @ (h * gain * v)
    result = result + 2 * cp * (h[:, None] * gain[None, :] * r[:, None] + gain[:, None] * h[None, :] * r[None, :])
    mixed = squared * (h * gain)[None, :] @ cp
    mixed = mixed * (h[:, None] * gain[None, :])
    result = result + 2 * (mixed + mixed.T)
    result = result + 2 * gain[:, None] * gain[None, :] * (cp * (h * h * v)[None, :] @ cp)
    tree31 = 3 * (q * gain ** 2 * v ** 2)[:, None] * gain[None, :] * cp + gain[:, None] ** 3 * q[None, :] * cp ** 3 + 6 * (h * h * gain * v ** 2)[:, None] * gain[None, :] * cp + 6 * (h * gain ** 2 * v)[:, None] * h[None, :] * cp ** 2
    repair = k31 - tree31
    result = result + (repair + repair.T)
    fnp.fill_diagonal(result, trace_diagonal)
    return (0.5 * (result + result.T)).astype(fnp.float32)

def lift_trace(trace_matrix):
    n = len(trace_matrix)
    result = trace_matrix.copy()
    fnp.fill_diagonal(result, fnp.diag(result) - fnp.asarray(fnp.trace(trace_matrix, dtype=fnp.float64) / (2 * (n + 2)), dtype=fnp.float32))
    return result / fnp.asarray(n + 4, dtype=fnp.float32)

def lift_covariance_trace(trace_matrix, covariance, iterations=3):
    """Solve the trace of sym6(B⊗C) by a contractive fixed-point iteration."""
    tr_c = fnp.trace(covariance, dtype=fnp.float64)
    tr_a = fnp.trace(trace_matrix, dtype=fnp.float64)
    result = (trace_matrix - covariance * fnp.asarray(tr_a / (2 * tr_c), dtype=fnp.float32)) / fnp.asarray(tr_c, dtype=fnp.float32)
    for _ in range(iterations):
        bc = result @ covariance
        tr_b = (tr_a - 4 * fnp.trace(bc, dtype=fnp.float64)) / (2 * tr_c)
        result = (trace_matrix - covariance * fnp.asarray(tr_b, dtype=fnp.float32) - 2 * (bc + bc.T)) / fnp.asarray(tr_c, dtype=fnp.float32)
    return (0.5 * (result + result.T)).astype(fnp.float32)

def inherited_trace(gram, transported_b, gain):
    d2 = gain * gain
    tg = fnp.sum(d2 * fnp.diag(gram), dtype=fnp.float64)
    tb = fnp.sum(d2 * fnp.diag(transported_b), dtype=fnp.float64)
    cross = gram * d2[None, :] @ transported_b
    result = gram * fnp.asarray(tb, dtype=fnp.float32) + transported_b * fnp.asarray(tg, dtype=fnp.float32) + 2 * (cross + cross.T)
    return result * gain[:, None] * gain[None, :]

def fourth_birth_trace_power(cp, sigma, phi, gain, hs, rs, degree=8, offdiag=True, edge_degree=2):
    """Return A_ij=sum_k kappa4(ReLU_i,ReLU_j,ReLU_k,ReLU_k).

    ``degree`` controls the two-neuron marginal Hermite expansion used for the
    exact-power kappa22 diagonal and kappa31 repairs. ``edge_degree`` truncates
    the total correlation degree of diagrams with three distinct neurons.
    Only one cubic matrix product is needed for edge_degree=2; three products
    are needed at degree3, with all b/c asymmetric terms reused by transpose.
    """
    n = len(sigma)
    mean = rs[1]
    variance = rs[2] - mean * mean
    gaussian_cov = covariance_hermite(cp, sigma, phi, gain, hs, degree)
    fnp.fill_diagonal(gaussian_cov, variance)
    h, g, cube = ({}, {}, {})
    centered_cubes, _ = _review_centered_cube_family(sigma, phi, gain, hs, rs, upto=max(degree, edge_degree))
    for order in range(1, max(degree, edge_degree) + 1):
        h[order] = derivative(1, order, sigma, phi, gain, hs, rs)
        g[order] = _review_centered_square(order, sigma, phi, gain, hs, rs)
        cube[order] = centered_cubes[order - 1]
    cp_powers = {0: fnp.ones_like(cp), 1: cp}
    for order in range(2, max(degree, edge_degree) + 1):
        cp_powers[order] = cp_powers[order - 1] * cp
    square_cov = fnp.zeros_like(cp)
    cube_cov = fnp.zeros_like(cp)
    for order in range(1, degree + 1):
        square_cov = square_cov + cp_powers[order] * fnp.outer(g[order], g[order]) / math.factorial(order)
        cube_cov = cube_cov + cp_powers[order] * fnp.outer(cube[order], h[order]) / math.factorial(order)
    k22 = square_cov - 2 * gaussian_cov * gaussian_cov
    k4diag = rs[4] - 4 * mean * rs[3] + 6 * mean * mean * rs[2] - 3 * mean ** 4 - 3 * variance * variance
    fnp.fill_diagonal(k22, k4diag)
    trace_diagonal = fnp.sum(k22, axis=1, dtype=fnp.float64).astype(fnp.float32)
    if not offdiag:
        directional_k31 = cube_cov - 3 * variance[:, None] * gaussian_cov
        fnp.fill_diagonal(directional_k31, k4diag)
        _derivative_cache['adjusted_gaussian_directional'] = (cp_powers[2], h[1], h[2], g[1], g[2], directional_k31, trace_diagonal, fnp.diag(cp))
        return fnp.diag(trace_diagonal)
    k31 = cube_cov - 3 * variance[:, None] * gaussian_cov
    fnp.fill_diagonal(k31, k4diag)
    result = fnp.zeros_like(cp)
    contracted_cache = {}
    diagonal = fnp.diag(cp)
    for total in range(2, edge_degree + 1):
        for a in range(total + 1):
            for b in range(total - a + 1):
                c = total - a - b
                if b < c or min(a + b, a + c, b + c) == 0:
                    continue
                central_coefficient = g[b + c]
                if a == 0:
                    central_coefficient = central_coefficient - 2 * h[b] * h[c]
                cache_key = (b, c, a == 0)
                if cache_key not in contracted_cache:
                    if c == 0:
                        contracted = (cp_powers[b] @ central_coefficient)[:, None]
                    else:
                        contracted = cp_powers[b] * central_coefficient[None, :] @ cp_powers[c].T
                    contracted_cache[cache_key] = contracted
                contracted = contracted_cache[cache_key]
                self_i = (diagonal ** b * central_coefficient)[:, None] * cp_powers[c]
                self_j = cp_powers[b] * (diagonal ** c * central_coefficient)[None, :]
                coefficient = cp_powers[a] * fnp.outer(h[a + b], h[a + c]) / (math.factorial(a) * math.factorial(b) * math.factorial(c))
                term = coefficient * (contracted - self_i - self_j)
                result = result + term
                if b != c:
                    result = result + term.T
    result = result + (k31 + k31.T)
    fnp.fill_diagonal(result, trace_diagonal)
    return (0.5 * (result + result.T)).astype(fnp.float32)

def star_slices(a, b):
    b2 = b * b
    diagonal = fnp.sum(a * b2 * b, axis=1)
    k31 = (3 * (a * b2 @ b.T) + b2 * b @ a.T) / 4
    ab_b2 = a * b @ b2.T
    k22 = (ab_b2 + ab_b2.T) / 2
    return (diagonal, k31, k22)

def star_trace(a, b):
    bnorm = fnp.sum(b * b, axis=0)
    abnorm = fnp.sum(a * b, axis=0)
    left = (a * bnorm[None, :] + b * abnorm[None, :]) @ b.T
    return (left + left.T) / 4

def repeat_slices(a, b, coefficients):
    a2, b2, ab = (a * a, b * b, a * b)
    diagonal = fnp.sum(a2 * b2 * coefficients[None, :], axis=1)
    k31 = (a2 * b * coefficients[None, :] @ b.T + a * b2 * coefficients[None, :] @ a.T) / 2
    left = a2 * coefficients[None, :] @ b2.T
    k22 = (left + left.T + 4 * (ab * coefficients[None, :]) @ ab.T) / 6
    return (diagonal, k31, k22)

def repeat_trace(a, b, coefficients):
    anorm = fnp.sum(a * a, axis=0)
    bnorm = fnp.sum(b * b, axis=0)
    abnorm = fnp.sum(a * b, axis=0)
    result = a * (coefficients * bnorm)[None, :] @ a.T
    _audit_local_transpose_b = b.T
    result = result + b * (coefficients * anorm)[None, :] @ _audit_local_transpose_b
    left = a * (coefficients * abnorm)[None, :] @ _audit_local_transpose_b
    return (result + 2 * (left + left.T)) / 6

def quadratic_slices(q, sign=1):
    """Slices of sign*(Q_ij Q_kl+Q_ik Q_jl+Q_il Q_jk)."""
    diagonal = fnp.diag(q)
    return (sign * 3 * diagonal * diagonal, sign * 3 * diagonal[:, None] * q, sign * (fnp.outer(diagonal, diagonal) + 2 * q * q))

def quadratic_trace(q, sign=1):
    return sign * (q * fnp.trace(q) + 2 * q @ q)

def gaussian_path_factors(cp, gain, h2, rank, offdiag=False, iterations=20, sketch_seed=0):
    """Spectral factors of the twelve connected Gaussian path diagrams.

    Their diagonal polynomial is 12*v.T@H@v, v_j=x_j*(L.T@x)_j,
    H_jk=h2_j*C_jk*h2_k, and L_ij=p_i*C_ij. Each spectral mode becomes
    a symmetric quadratic-form square and transports through W exactly.
    With ``offdiag``, coincident internal vertices and endpoint vertices
    are excluded for composition with exact repeated-power births.
    """
    n = len(gain)
    rank = min(rank, n)
    h = cp * h2[:, None] * h2[None, :]
    leaf = cp * gain[:, None]
    if offdiag:
        h = h.copy()
        leaf = leaf.copy()
        fnp.fill_diagonal(h, 0)
        fnp.fill_diagonal(leaf, 0)
    if rank == n:
        eigenvalues, basis = fnp.linalg.eigh(h.astype(fnp.float64))
    else:
        sketch = fnp.random.default_rng(sketch_seed).standard_normal((n, min(n, rank + 8))).astype(fnp.float32)
        basis = fnp.linalg.qr(sketch, mode='reduced')[0]
        for _ in range(iterations):
            basis = fnp.linalg.qr(h @ basis, mode='reduced')[0]
        small = basis.T @ h @ basis
        eigenvalues, rotation = fnp.linalg.eigh((small + small.T) / 2)
        basis = basis @ rotation
    selected = fnp.argsort(fnp.abs(eigenvalues))[-rank:]
    eigenvalues, basis = (eigenvalues[selected], basis[:, selected])
    factors = []
    for value, vector in zip(eigenvalues, basis.T):
        left = leaf * vector[None, :]
        q = (fnp.sqrt(abs(value)) * (left + left.T)).astype(fnp.float32)
        factors.append((q, fnp.where(value >= 0, fnp.asarray(1.0, dtype=fnp.float32), fnp.asarray(-1.0, dtype=fnp.float32))))
    diagonal_internal = fnp.sum(basis * basis * eigenvalues[None, :], axis=1).astype(fnp.float32)
    return (factors, diagonal_internal)

def diagonal_pair_factors(target, existing, rank, iterations=20, sketch_seed=0):
    """Repair Gaussian κ22 using signed diagonal quadratic-form squares.

    The diagonal of the repair matrix is zero because marginal κ4 is already
    exact. A truncated spectral repair invents a diagonal; an explicit e_i^4
    counterterm cancels it while preserving every retained off-diagonal mode.
    """
    difference = (target - existing).astype(fnp.float32)
    fnp.fill_diagonal(difference, 0)
    difference = (difference + difference.T) / 2
    n = len(difference)
    rank = min(rank, n)
    if rank == n:
        values, basis = fnp.linalg.eigh(difference.astype(fnp.float64))
    else:
        sketch = fnp.random.default_rng(sketch_seed).standard_normal((n, min(n, rank + 8))).astype(fnp.float32)
        basis = fnp.linalg.qr(sketch, mode='reduced')[0]
        for _ in range(iterations):
            basis = fnp.linalg.qr(difference @ basis, mode='reduced')[0]
        small = basis.T @ difference @ basis
        values, rotation = fnp.linalg.eigh((small + small.T) / 2)
        basis = basis @ rotation
    selected = fnp.argsort(fnp.abs(values))[-rank:]
    values, basis = (values[selected], basis[:, selected])
    factors = [(fnp.diag(fnp.sqrt(abs(value)) * vector).astype(fnp.float32), fnp.where(value >= 0, fnp.asarray(1.0, dtype=fnp.float32), fnp.asarray(-1.0, dtype=fnp.float32))) for value, vector in zip(values, basis.T)]
    invented_diagonal = fnp.sum(basis * basis * values[None, :], axis=1).astype(fnp.float32)
    counter = (fnp.diag(-3 * invented_diagonal), fnp.eye(n, dtype=fnp.float32))
    return (factors, counter)

def gaussian_power_slices(cp, sigma, phi, gain, hs, rs, degree, derivative, covariance_hermite):
    mean = rs[1]
    variance = rs[2] - mean * mean
    cov = covariance_hermite(cp, sigma, phi, gain, hs, degree)
    fnp.fill_diagonal(cov, variance)
    rho = fnp.clip(cp / (sigma[:, None] * sigma[None, :]), -1, 1)
    fnp.fill_diagonal(rho, 0)
    power = fnp.ones_like(cp)
    square_cov = fnp.zeros_like(cp)
    cube_cov = fnp.zeros_like(cp)
    for k in range(1, degree + 1):
        power = power * rho
        h = derivative(1, k, sigma, phi, gain, hs, rs)
        g = derivative(2, k, sigma, phi, gain, hs, rs) - 2 * mean * h
        cube = derivative(3, k, sigma, phi, gain, hs, rs) - 3 * mean * derivative(2, k, sigma, phi, gain, hs, rs) + 3 * mean * mean * h
        scale = sigma ** k
        square_cov = square_cov + power * fnp.outer(g * scale, g * scale) / math.factorial(k)
        cube_cov = cube_cov + power * fnp.outer(cube * scale, h * scale) / math.factorial(k)
    diagonal = rs[4] - 4 * mean * rs[3] + 6 * mean * mean * rs[2] - 3 * mean ** 4 - 3 * variance * variance
    k31 = cube_cov - 3 * variance[:, None] * cov
    k22 = square_cov - 2 * cov * cov
    fnp.fill_diagonal(k31, diagonal)
    fnp.fill_diagonal(k22, diagonal)
    return (diagonal, k31, k22)

def gaussian_fourth_factors(cp, sigma, phi, gain, hs, rs, degree, derivative, covariance_hermite, mode='power'):
    """Gaussian-born stars, repeated-power vertices, and exact κ31 repairs.

    ``tree`` preserves the four ordinary Hermite star diagrams. ``power``
    additionally preserves the exact leading i,i,j,k centered-square vertex,
    whose coefficient is E[g_i''] - 2 E[f_i']**2, and exact Gaussian i,i,i,j
    slices. The remaining fourth tensor is supplied by a trace residual.
    """
    n = len(sigma)
    h2 = phi / sigma
    h3 = -hs[1] * phi / sigma ** 2
    leaf = gain[:, None] * cp
    star_a = fnp.diag(4 * h3).astype(fnp.float32)
    repeat = None
    if mode == 'tree':
        return ((star_a, leaf.astype(fnp.float32)), repeat)
    if mode not in ('power', 'repeat'):
        raise ValueError(mode)
    fnp.fill_diagonal(leaf, 0)
    g2 = 2 * gain - 2 * rs[1] * h2
    repeat = (fnp.eye(n, dtype=fnp.float32), leaf.copy(), (6 * (g2 - 2 * gain * gain)).astype(fnp.float32))
    diagonal, k31, _ = gaussian_power_slices(cp, sigma, phi, gain, hs, rs, degree, derivative, covariance_hermite)
    existing = h3[None, :] * leaf ** 3 if mode == 'power' else 0
    repair = (4 * (k31 - existing).T).astype(fnp.float32)
    fnp.fill_diagonal(repair, diagonal)
    if mode == 'power':
        return ((fnp.concatenate([star_a, repair], axis=1), fnp.concatenate([leaf, fnp.eye(n, dtype=fnp.float32)], axis=1)), repeat)
    return ((repair, fnp.eye(n, dtype=fnp.float32)), repeat)

def correlated_third_response(cp, skew, pair, di, dj, order):
    """Connected Edgeworth κ3 response with Gaussian derivative correlations.

    E[f_i^(a)g_j^(b)] = sum_r Cij^r/r! d_(a+r)f_i*d_(b+r)g_j.
    Marginal r=0 terms cancel in covariance, so they begin at r=1.
    """
    response = fnp.zeros_like(cp)
    power = fnp.ones_like(cp)
    factorial = 1.0
    for r in range(order + 1):
        if r:
            power = power * cp
            factorial = factorial * r
        coefficient = power / factorial
        response = response + 0.5 * pair * coefficient * di[r + 2][:, None] * dj[r + 1][None, :]
        response = response + 0.5 * pair.T * coefficient * di[r + 1][:, None] * dj[r + 2][None, :]
    power = fnp.ones_like(cp)
    factorial = 1.0
    for r in range(1, max(1, order) + 1):
        power = power * cp
        factorial = factorial * r
        coefficient = power / factorial
        response = response + coefficient * skew[:, None] * di[r + 3][:, None] * dj[r][None, :] / 6
        response = response + coefficient * skew[None, :] * di[r][:, None] * dj[r + 3][None, :] / 6
    return response

def second_third_response(cp, skew, pair, di, dj):
    """Second Edgeworth κ3² covariance: mixed r=0 and marginal r=1 terms."""
    a = skew[:, None] / 6
    d = skew[None, :] / 6
    b, c = (pair / 2, pair.T / 2)
    response = a * b * di[5][:, None] * dj[1][None, :]
    response = response + (a * c + b * b / 2) * di[4][:, None] * dj[2][None, :]
    response = response + b * c * di[3][:, None] * dj[3][None, :]
    response = response + (b * d + c * c / 2) * di[2][:, None] * dj[4][None, :]
    response = response + c * d * di[1][:, None] * dj[5][None, :]
    response = response + cp * (a * a / 2 * di[7][:, None] * dj[1][None, :] + a * d * di[4][:, None] * dj[4][None, :] + d * d / 2 * di[1][:, None] * dj[7][None, :])
    return response

def correlated_fourth_response(cp, kurt, k31, k22, di, dj, order):
    """Connected Edgeworth κ4 response, including derivative correlations."""
    response = fnp.zeros_like(cp)
    power = fnp.ones_like(cp)
    factorial = 1.0
    for r in range(order + 1):
        if r:
            power = power * cp
            factorial = factorial * r
        coefficient = power / factorial
        response = response + coefficient * k31 * di[r + 3][:, None] * dj[r + 1][None, :] / 6
        response = response + coefficient * k31.T * di[r + 1][:, None] * dj[r + 3][None, :] / 6
        response = response + coefficient * k22 * di[r + 2][:, None] * dj[r + 2][None, :] / 4
    power = fnp.ones_like(cp)
    factorial = 1.0
    for r in range(1, max(1, order) + 1):
        power = power * cp
        factorial = factorial * r
        coefficient = power / factorial
        response = response + coefficient * kurt[:, None] * di[r + 4][:, None] * dj[r][None, :] / 24
        response = response + coefficient * kurt[None, :] * di[r][:, None] * dj[r + 4][None, :] / 24
    return response

def correlated_fourth_square(cp, gaussian_cov, pair_gaussian, gain, rs, sigma, phi, hs, kurt, k31, k22, order, d1, d2, covariance_fourth):
    delta_mean = kurt * d1[4] / 24
    delta = correlated_fourth_response(cp, kurt, k31, k22, d2, d2, order)
    centering = -2 * delta_mean[:, None] * pair_gaussian.T
    delta = delta + (centering + centering.T)
    delta = delta - 4 * gaussian_cov * covariance_fourth
    delta = delta - k22 * gain[:, None] ** 2 * gain[None, :] ** 2
    corrected = [rs[0]] + [rs[p] + kurt * derivative(p, 4, sigma, phi, gain, hs, rs) / 24 for p in range(1, 5)]
    mu = corrected[1]
    var = corrected[2] - mu ** 2
    k4 = corrected[4] - 4 * mu * corrected[3] + 6 * mu ** 2 * corrected[2] - 3 * mu ** 4 - 3 * var ** 2
    m0 = rs[1]
    v0 = rs[2] - m0 ** 2
    k40 = rs[4] - 4 * m0 * rs[3] + 6 * m0 ** 2 * rs[2] - 3 * m0 ** 4 - 3 * v0 ** 2
    fnp.fill_diagonal(delta, k4 - k40 - gain ** 4 * kurt)
    return fnp.sum(delta, axis=1, dtype=fnp.float64).astype(fnp.float32)

def triangle_factors(cp, h2, rank=1, iterations=16):
    """Low-rank CP representation of h2_i h2_j h2_k Cij Cjk Cki.

    The triangle is exact for the leading-rank covariance approximation.
    The separate exact third trace absorbs its repeated-index discrepancy.
    """
    n = len(h2)
    indices = fnp.arange(n, dtype=fnp.float32)[:, None]
    frequencies = fnp.arange(1, rank + 1, dtype=fnp.float32)[None, :]
    q = fnp.cos(indices * frequencies * fnp.asarray(0.173, dtype=fnp.float32))
    q, _ = fnp.linalg.qr(q, mode='reduced')
    for _ in range(iterations):
        q, _ = fnp.linalg.qr(cp @ q, mode='reduced')
    eigenvalues, rotation = fnp.linalg.eigh(q.T @ cp @ q)
    order = fnp.argsort(eigenvalues)[::-1]
    values = fnp.maximum(eigenvalues[order], fnp.asarray(0, dtype=fnp.float32))
    vectors = q @ rotation[:, order]
    af, bf = ([], [])
    for r in range(rank):
        v = values[r] * h2 * vectors[:, r] ** 2
        af.append(v)
        bf.append(v)
    for r in range(rank):
        for s in range(rank):
            if r == s:
                continue
            af.append(3 * values[r] * h2 * vectors[:, r] ** 2)
            bf.append(fnp.sqrt(values[r] * values[s]) * h2 * vectors[:, r] * vectors[:, s])
    for r in range(rank):
        for s in range(r + 1, rank):
            for t in range(s + 1, rank):
                a = fnp.sqrt(values[r] * values[t]) * h2 * vectors[:, r] * vectors[:, t]
                b = fnp.sqrt(values[r] * values[s]) * h2 * vectors[:, r] * vectors[:, s]
                c = fnp.sqrt(values[s] * values[t]) * h2 * vectors[:, s] * vectors[:, t]
                for v in (a + b + c, a - b - c, -a + b - c, -a - b + c):
                    af.append(v / 4)
                    bf.append(v)
    return (fnp.column_stack(af).astype(fnp.float32), fnp.column_stack(bf).astype(fnp.float32))

def correlated_square_feedback(cp, gaussian_cov, pair_gaussian, gain, rs, sigma, phi, hs, skew, pair, order, d1, d2, covariance_third):
    delta_mean = skew * d1[3] / 6
    delta = correlated_third_response(cp, skew, pair, d2, d2, order)
    centering = -2 * delta_mean[:, None] * pair_gaussian.T
    delta = delta + (centering + centering.T)
    delta = delta - 4 * gaussian_cov * covariance_third
    corrected = [rs[0]] + [rs[p] + skew * derivative(p, 3, sigma, phi, gain, hs, rs) / 6 for p in range(1, 5)]
    mean = corrected[1]
    var = corrected[2] - mean ** 2
    c4 = corrected[4] - 4 * mean * corrected[3] + 6 * mean ** 2 * corrected[2] - 3 * mean ** 4
    m0 = rs[1]
    v0 = rs[2] - m0 ** 2
    c40 = rs[4] - 4 * m0 * rs[3] + 6 * m0 ** 2 * rs[2] - 3 * m0 ** 4
    fnp.fill_diagonal(delta, c4 - 3 * var ** 2 - (c40 - 3 * v0 ** 2))
    return fnp.sum(delta, axis=1, dtype=fnp.float64).astype(fnp.float32)

def third_trace_source(gram, third_linear, pre_banks, gain, rs, sigma, phi, hs, cp, gaussian_cov, pair_gaussian, skew, pair, covariance_third, bank_limit=16, repair_order=1):
    """Leading distinct-index κ3 contribution to the fourth trace matrix.

    Contract the third tensor with the first Gaussian derivative of the
    centered square.  Exact bivariate power responses replace k=i,j; diagonal
    κ22 feedback is owned by the separate square-cumulant routine.
    """
    n = len(gain)
    mu = rs[1]
    g1 = 2 * mu * (1 - gain)
    gram_g1 = gram @ g1
    contracted = gram * fnp.dot(third_linear, g1) + gram_g1[:, None] * third_linear[None, :] + third_linear[:, None] * gram_g1[None, :]
    retained = pre_banks[-bank_limit:] if bank_limit else []
    discarded = pre_banks[:max(0, len(pre_banks) - bank_limit)]
    if retained:
        if len(retained) == len(pre_banks) and 'efficiency_full_cp_pre' in _derivative_cache:
            af, bf = _derivative_cache['efficiency_full_cp_pre']
        else:
            af = fnp.concatenate([a for a, b in retained], axis=1)
            bf = fnp.concatenate([b for a, b in retained], axis=1)
        _audit_local_transpose_bf = bf.T
        left = af * (_audit_local_transpose_bf @ g1)[None, :] + 0.5 * bf * (af.T @ g1)[None, :]
        ab = left @ _audit_local_transpose_bf
        fnp.add(contracted, (ab + ab.T) / 3, out=contracted)
    for af, bf in discarded:
        trace = (af @ fnp.sum(bf * bf, axis=0) + 2 * bf @ fnp.sum(af * bf, axis=0)) / 3
        lifted = trace / (n + 2)
        fnp.add(contracted, g1[:, None] * lifted[None, :] + lifted[:, None] * g1[None, :], out=contracted)
        fnp.fill_diagonal(contracted, fnp.diag(contracted) + fnp.dot(lifted, g1))
    _derivative_cache['efficiency_cubic_g1'] = contracted
    result = contracted * gain[:, None] * gain[None, :]
    repeated = gain[:, None] * gain[None, :] * (pair * g1[:, None] + pair.T * g1[None, :])
    result = result - repeated
    highest = max(1, repair_order) + 3
    d1 = [rs[1]] + [derivative(1, k, sigma, phi, gain, hs, rs) for k in range(1, highest + 1)]
    cube = [rs[3] - 3 * mu * rs[2] + 2 * mu ** 3]
    cube = cube + [derivative(3, k, sigma, phi, gain, hs, rs) - 3 * mu * derivative(2, k, sigma, phi, gain, hs, rs) + 3 * mu ** 2 * d1[k] for k in range(1, highest + 1)]
    k31_delta = correlated_third_response(cp, skew, pair, cube, d1, repair_order)
    delta_mean = skew * d1[3] / 6
    delta_variance = skew * (derivative(2, 3, sigma, phi, gain, hs, rs) - 2 * mu * d1[3]) / 6
    variance = rs[2] - mu ** 2
    k31_delta = k31_delta - 3 * delta_mean[:, None] * pair_gaussian
    k31_delta = k31_delta - 3 * delta_variance[:, None] * gaussian_cov
    k31_delta = k31_delta - 3 * variance[:, None] * covariance_third
    result = result + (k31_delta + k31_delta.T)
    fnp.fill_diagonal(result, 0)
    return (0.5 * (result + result.T)).astype(fnp.float32)

def third_covariance_trace_source(cp, gram, third_linear, pre_banks, gain, h2, skew, pair):
    """Distinct-index covariance×κ3 diagrams omitted by the g1 contraction.

    A quadratic response at vertex l contributes h2_l*p_i*p_j*p_k times
    C_il*T3_jkl+C_jl*T3_ikl+C_kl*T3_ijl. The center-k variance term is
    already represented by g1_k*T3_ijk. This adds the other contractions,
    excludes k=i,j, and leaves all exact power-slice repairs unchanged.
    """
    p2 = gain * gain
    weighted_cp = cp * p2[None, :]
    contracted = gram * (cp @ (p2 * third_linear))[:, None]
    contracted = contracted + fnp.sum(weighted_cp * gram, axis=1)[:, None] * third_linear[None, :]
    contracted = contracted + third_linear[:, None] * (weighted_cp @ gram)
    for af, bf in pre_banks:
        cb, ca = (weighted_cp @ bf, weighted_cp @ af)
        contracted = contracted + ((af * cb + bf * ca) @ bf.T + bf * cb @ af.T) / 3
    weighted_trace = p2 @ pair
    center = h2[:, None] * gain[None, :] * (cp * weighted_trace[:, None] + 2 * contracted)
    variance = fnp.diag(cp)
    excluded = cp * (p2 * skew)[:, None]
    excluded = excluded + 3 * cp * pair.T * p2[None, :]
    excluded = excluded + 2 * (p2 * variance)[:, None] * pair
    center = center - h2[:, None] * gain[None, :] * excluded
    power_vertex = h2 * gain
    repeated = cp * power_vertex[None, :] @ pair
    repeated = repeated - (variance * power_vertex)[:, None] * pair
    repeated = repeated - cp * (power_vertex * skew)[None, :]
    result = center + center.T
    result = result + 2 * gain[:, None] * gain[None, :] * (repeated + repeated.T)
    fnp.fill_diagonal(result, 0)
    return (0.5 * (result + result.T)).astype(fnp.float32)

def square_cumulant_feedback(cp, gaussian_cov, gain, rs, corrected, sigma, phi, hs, skew_pre, pair_pre, kurt_pre, k31, k22, covariance_delta, include_fourth=True, consistent_third_only=False, third_covariance=None):
    """First non-Gaussian power response of κ(y_i,y_i,y_j,y_j).

    Centered-square coefficients include the shift of the post-ReLU mean.
    Subtract the transported D_i² D_j² κ22 term, which inherited_trace already
    carries; the remaining piece is nonlinear fourth-cumulant birth.
    """
    mu = rs[1]
    h2 = phi / sigma
    h3 = derivative(1, 3, sigma, phi, gain, hs, rs)
    h4 = derivative(1, 4, sigma, phi, gain, hs, rs)
    h5 = derivative(1, 5, sigma, phi, gain, hs, rs)
    if not include_fourth and consistent_third_only:
        corrected = [rs[0]] + [rs[p] + skew_pre * derivative(p, 3, sigma, phi, gain, hs, rs) / 6 for p in range(1, 5)]
        fourth_delta = k31 * h3[:, None] * gain[None, :] / 6
        fourth_delta = fourth_delta + fourth_delta.T + k22 * h2[:, None] * h2[None, :] / 4
        fourth_gain = kurt_pre * h5 / 24
        fourth_delta = fourth_delta + cp * (fourth_gain[:, None] * gain[None, :] + gain[:, None] * fourth_gain[None, :])
        covariance_delta = covariance_delta - fourth_delta if third_covariance is None else third_covariance
    g1 = 2 * mu * (1 - gain)
    g2 = 2 * gain - 2 * mu * h2
    g3 = 2 * h2 - 2 * mu * h3
    g4 = 2 * h3 - 2 * mu * h4
    g5 = 2 * h4 - 2 * mu * h5
    delta = 0.5 * pair_pre * g2[:, None] * g1[None, :]
    delta = delta + delta.T
    gain_delta = skew_pre * g4 / 6
    if include_fourth:
        term = k31 * g3[:, None] * g1[None, :] / 6
        delta = delta + (term + term.T)
        delta = delta + k22 * g2[:, None] * g2[None, :] / 4
        gain_delta = gain_delta + kurt_pre * g5 / 24
        delta = delta - k22 * (gain * gain)[:, None] * (gain * gain)[None, :]
    gain_delta = gain_delta - 2 * (corrected[1] - mu) * gain
    delta = delta + cp * (gain_delta[:, None] * g1[None, :] + g1[:, None] * gain_delta[None, :])
    delta = delta - 4 * gaussian_cov * covariance_delta
    post_mean = corrected[1]
    post_var = corrected[2] - post_mean ** 2
    c4 = corrected[4] - 4 * post_mean * corrected[3] + 6 * post_mean ** 2 * corrected[2] - 3 * post_mean ** 4
    gaussian_var = rs[2] - mu ** 2
    gaussian_c4 = rs[4] - 4 * mu * rs[3] + 6 * mu ** 2 * rs[2] - 3 * mu ** 4
    diag_delta = c4 - 3 * post_var ** 2 - (gaussian_c4 - 3 * gaussian_var ** 2)
    if include_fourth:
        diag_delta = diag_delta - gain ** 4 * kurt_pre
    fnp.fill_diagonal(delta, diag_delta)
    return fnp.sum(delta, axis=1, dtype=fnp.float64).astype(fnp.float32)

def _direct_source_state(cp, h2, p, a, b, strength=1.0):
    return dict(edge=fnp.asarray(cp) * p[None, :] * strength, da=h2[:, None] * a, db=h2[:, None] * b, va=p[:, None] * a, vb=p[:, None] * b, transport=fnp.eye(len(p), dtype=cp.dtype), identity=True)

def _direct_moved_state(state, weights, gain=None):
    result = dict(state)
    result['transport'] = state['transport'] @ weights
    if gain is not None:
        result['transport'] = result['transport'] * gain[None, :]
    result['identity'] = False
    return result

def _direct_context(state):
    t = state['transport']
    identity = state['identity']
    h = state['edge'] if identity else state['edge'] @ t
    _audit_local_transpose_t = t.T
    va = state['va'] if identity else _audit_local_transpose_t @ state['va']
    vb = state['vb'] if identity else _audit_local_transpose_t @ state['vb']
    if identity:
        qa = fnp.diag(h)[:, None] * state['da']
        qb = fnp.diag(h)[:, None] * state['db']
    else:
        diagonal_map = (t * h).T
        qa = diagonal_map @ state['da']
        qb = diagonal_map @ state['db']
    return dict(t=t, h=h, da=state['da'], db=state['db'], va=va, vb=vb, qa=qa, qb=qb, identity=identity)

def _direct__row_weighted_q(ctx, d, weights):
    """Matrix sum_r Qr_ij weights_ir without materializing Qr."""
    t, h = (ctx['t'], ctx['h'])
    matrix = weights @ d.T
    if ctx['identity']:
        return (fnp.diag(matrix)[:, None] * h + matrix * h.T) / 2
    return (matrix * t.T @ h + matrix * h.T @ t) / 2

def _direct__shared_weighted_q(ctx, d, weights):
    """Matrix sum_r weights_r Qr."""
    t, h = (ctx['t'], ctx['h'])
    field = d @ weights
    left = field[:, None] * h if ctx['identity'] else t.T * field[None, :] @ h
    return (left + left.T) / 2

def _direct_diagonal_and_k31(ctx):
    va, vb, qa, qb = (ctx['va'], ctx['vb'], ctx['qa'], ctx['qb'])
    diagonal = fnp.sum(4 * qa * vb * vb + 8 * qb * va * vb, axis=1)
    k31 = 2 * _direct__row_weighted_q(ctx, ctx['da'], vb * vb)
    _audit_local_transpose_vb = vb.T
    k31 = k31 + 2 * (qa * vb) @ _audit_local_transpose_vb
    k31 = k31 + 4 * _direct__row_weighted_q(ctx, ctx['db'], va * vb)
    k31 = k31 + 2 * (qb * vb @ va.T + qb * va @ _audit_local_transpose_vb)
    fnp.fill_diagonal(k31, diagonal)
    return (diagonal, k31)

def _direct__weighted_q_vectors(ctx, weight):
    """Qa_r(weight*vb_r), Qb_r(weight*va_r), Qb_r(weight*vb_r)."""
    t, h, da, db = (ctx['t'], ctx['h'], ctx['da'], ctx['db'])
    wa = ctx['va'] * weight[:, None]
    wb = ctx['vb'] * weight[:, None]
    ha, hb = (h @ wa, h @ wb)
    if ctx['identity']:
        _audit_local_transpose_h = h.T
        sa = (da * hb + _audit_local_transpose_h @ (da * wb)) / 2
        sba = (db * ha + _audit_local_transpose_h @ (db * wa)) / 2
        sbb = (db * hb + _audit_local_transpose_h @ (db * wb)) / 2
    else:
        ta, tb = (t @ wa, t @ wb)
        _audit_local_transpose_h = h.T
        _audit_local_transpose_t = t.T
        sa = (_audit_local_transpose_t @ (da * hb) + _audit_local_transpose_h @ (da * tb)) / 2
        sba = (_audit_local_transpose_t @ (db * ha) + _audit_local_transpose_h @ (db * ta)) / 2
        sbb = (_audit_local_transpose_t @ (db * hb) + _audit_local_transpose_h @ (db * tb)) / 2
    return (sa, sba, sbb)

def _direct_weighted_trace(ctx, weight):
    """Return sum_k weight_k T4_ijkk, exactly, in O(n²*CPcols)."""
    va, vb, qa, qb = (ctx['va'], ctx['vb'], ctx['qa'], ctx['qb'])
    sa, sba, sbb = _direct__weighted_q_vectors(ctx, weight)
    trqa, trqb = (weight @ qa, weight @ qb)
    vvb = weight @ (vb * vb)
    vab = weight @ (va * vb)
    _audit_local_transpose_vb = vb.T
    tracea = vb * trqa[None, :] @ _audit_local_transpose_vb
    tracea = tracea + 2 * (sa @ _audit_local_transpose_vb + vb @ sa.T)
    tracea = tracea + _direct__shared_weighted_q(ctx, ctx['da'], vvb)
    _audit_local_transpose_va = va.T
    traceb = (va * trqb[None, :] @ _audit_local_transpose_vb + vb * trqb[None, :] @ _audit_local_transpose_va) / 2
    traceb = traceb + (sba @ _audit_local_transpose_vb + vb @ sba.T + sbb @ _audit_local_transpose_va + va @ sbb.T)
    traceb = traceb + _direct__shared_weighted_q(ctx, ctx['db'], vab)
    result = 2 / 3 * tracea + 4 / 3 * traceb
    return (result + result.T) / 2

def _direct_pair_matvec(ctx, weight):
    """Return the complete exact kappa22@weight in O(n²*CPcols)."""
    va, vb, qa, qb = (ctx['va'], ctx['vb'], ctx['qa'], ctx['qb'])
    sa, sba, sbb = _direct__weighted_q_vectors(ctx, weight)
    first = qa @ ((vb * vb).T @ weight) + vb * vb @ (qa.T @ weight)
    first = first + 4 * fnp.sum(vb * sa, axis=1)
    second = qb @ ((va * vb).T @ weight) + va * vb @ (qb.T @ weight)
    second = second + 2 * fnp.sum(va * sbb + vb * sba, axis=1)
    return 2 / 3 * first + 4 / 3 * second

def _direct_functional_pair_matrix(weights, responses, diagonal, iterations=8):
    """Symmetric functional approximation with exact diagonal. Eight projections restore the specified exact matrix-vector responses; research-only norm diagnostics are omitted."""
    weights = fnp.asarray(weights, fnp.float64)
    responses = fnp.asarray(responses, fnp.float64)
    q, r = fnp.linalg.qr(weights, mode='reduced')
    y = fnp.linalg.solve(r.T, responses.T).T
    _audit_local_transpose_q = q.T
    core = _audit_local_transpose_q @ y
    core = (core + core.T) / 2
    matrix = q @ y.T + y @ _audit_local_transpose_q - q @ core @ _audit_local_transpose_q
    for _ in range(iterations):
        fnp.fill_diagonal(matrix, diagonal)
        error = y - matrix @ q
        _audit_local_transpose_error = error.T
        _audit_local_transpose_q = _audit_local_transpose_q
        correction = q @ _audit_local_transpose_error + error @ _audit_local_transpose_q - q @ ((_audit_local_transpose_q @ error + _audit_local_transpose_error @ q) / 2) @ _audit_local_transpose_q
        matrix = matrix + correction
    fnp.fill_diagonal(matrix, diagonal)
    return (matrix, {})

def weighted_third(gram, vector, banks, weights):
    gw = gram @ weights
    result = gram * fnp.dot(vector, weights)
    result = result + (gw[:, None] * vector[None, :] + vector[:, None] * gw[None, :])
    for a, b in banks:
        _audit_transpose_b = b.T
        left = a * (_audit_transpose_b @ weights)[None, :] + 0.5 * b * (a.T @ weights)[None, :]
        ab = left @ _audit_transpose_b
        result = result + (ab + ab.T) / 3
    return result

def covariance_weighted_third(cp, gram, vector, banks, weights):
    cw = cp * weights[None, :]
    result = gram * (cp @ (weights * vector))[:, None]
    result = result + fnp.sum(cw * gram, axis=1)[:, None] * vector[None, :]
    result = result + vector[:, None] * (cw @ gram)
    for a, b in banks:
        cb, ca = (cw @ b, cw @ a)
        result = result + ((a * cb + b * ca) @ b.T + b * cb @ a.T) / 3
    return result

def third_edge_fourth_trace(cp, gram, vector, banks, gain, h2, h3, g1, g2, g3, skew, pair):
    """Sum_k edge-order-one response of κ4(F_i,F_j,F_k,F_k).

    ``pair[i,j]`` denotes κ3(X_i,X_i,X_j). The three-variable
    expansion applies Gaussian derivative correlations to the centered-square
    vertex before cumulant centralization, rather than approximating that
    vertex by two independent linear outputs.
    """
    c = cp.copy()
    fnp.fill_diagonal(c, 0)
    if 'efficiency_cubic_g1' in _derivative_cache:
        trace1 = _derivative_cache['efficiency_cubic_g1']
    else:
        trace1 = weighted_third(gram, vector, banks, g1)
    _audit_local_transpose_pair = pair.T
    trace1 = trace1 - (g1[:, None] * pair + g1[None, :] * _audit_local_transpose_pair)
    row1 = pair @ g1 - g1 * skew
    row_excluded = row1[:, None] - pair * g1[None, :]
    col2 = g2 @ pair - g2 * skew
    col_excluded = col2[:, None] - _audit_local_transpose_pair * g2[None, :]
    term = c * (0.5 * h3[:, None] * gain[None, :] * row_excluded + 0.5 * gain[:, None] * h3[None, :] * row_excluded.T + h2[:, None] * h2[None, :] * trace1 + 0.5 * h2[:, None] * gain[None, :] * col_excluded + 0.5 * gain[:, None] * h2[None, :] * col_excluded.T)
    q = (c @ g1)[:, None] - c * g1[None, :]
    center = 0.5 * q * (h3[:, None] * gain[None, :] * pair + h2[:, None] * h2[None, :] * _audit_local_transpose_pair)
    if banks:
        full_a = fnp.concatenate([a for a, b in banks], axis=1)
        full_b = fnp.concatenate([b for a, b in banks], axis=1)
        basis = _efficiency_covariance_basis(gram, 8, sketch_seed=2026)
        projected_a, projected_b = _efficiency_project_b_slots(full_a, full_b, basis)
        projected_banks = [(projected_a, projected_b)]
    else:
        projected_banks = []
    contraction = covariance_weighted_third(c, gram, vector, projected_banks, g2)
    contraction = contraction - c * g2[None, :] * _audit_local_transpose_pair
    center = center + h2[:, None] * gain[None, :] * contraction
    w3 = g3 / 2 - gain * h2
    w4 = g2 / 2 - gain * gain
    power3 = c * w3[None, :] @ pair - c * (w3 * skew)[None, :]
    power4 = c * w4[None, :] @ _audit_local_transpose_pair - c * (w4 * skew)[None, :]
    center = center + gain[:, None] * gain[None, :] * power3
    center = center + gain[:, None] * h2[None, :] * power4
    term = term + (center + center.T)
    fnp.fill_diagonal(term, 0)
    return 0.5 * (term + term.T)

def _predict(weights, degree=4, horizon=16, cp_family='path', fourth_mode='full', feedback=True, include_fourth_feedback=True, consistent_third_only=False, return_state=False, correlation_order=1, third_matrix_banks=16, gaussian_trace_edge_degree=3, second_third=None, fourth_metric='covariance', third_metric='covariance', fourth_degree=None, correlation_parts='pair', teacher_means=None, fourth_correlation_order=-1, fourth_correlation_parts='all', triangle_rank=0, birth_mode='effective', include_gaussian_edge=True, angular_input=True, fourth_factor_horizon=16, fourth_factor_mode='tree', third_covariance_feedback=False, fourth_path_rank=0, fourth_path_horizon=None, fourth_lift_iterations=3, coherent_pair_strength=0.5, fourth_pair_rank=0, fourth_pair_horizon=4, direct_horizon=0, direct_banks_limit=None, direct_pair_rank=2, direct_strength=1.0, sketch_seed=0, readout_coefficients=None, activation_cp_threshold=0.01, activation_cp_upper=0.99):
    global _adjusted_pooled_cp_engine
    _adjusted_pooled_cp_engine = _adjusted_pooled_cp_engine_for_width(weights[0].shape[0])
    if coherent_pair_strength != 0 or triangle_rank != 0 or fourth_factor_horizon != 0 or (fourth_path_rank != 0) or (direct_horizon != 0):
        raise ValueError('This source is specialized to lambda0/triangle0/starH1/path0/direct0.')
    n = weights[0].shape[0]
    correction = fnp.zeros(n, dtype=fnp.float64)
    mean = fnp.zeros(n, fnp.float32)
    cov = fnp.eye(n, dtype=fnp.float32)
    third_a = fnp.zeros(n, fnp.float32)
    fourth_scalar = 0.0
    radius_variance = 2 * math.exp(2 * (math.lgamma((n + 1) / 2) - math.lgamma(n / 2))) / n
    cov = cov * fnp.asarray(radius_variance, dtype=fnp.float32)
    fourth_scalar = -1.0 / (n + 2)
    fourth_diagonal = fnp.zeros(n, dtype=fnp.float32)
    banks = []
    cp_active_rows = fnp.arange(n)
    cp_birth_rows = fnp.arange(n)
    fourth_factor_banks = []
    fourth_path_banks = []
    fourth_pair_banks = []
    direct_banks = []
    path_horizon = fourth_factor_horizon
    rows, diagnostics, state_history = ([], [], [])
    for layer, w in enumerate(weights):
        final_pair_only = layer == len(weights) - 1 and (not return_state)
        _derivative_cache.clear()
        _review_vector_cache.clear()
        _covariance_cache.clear()
        w = fnp.asarray(w, dtype=fnp.float32)
        wt = w.T
        mu = mean @ w
        if teacher_means is not None and layer:
            mu = teacher_means[layer - 1] @ w
        cp = fnp.einsum('ij,ia,jb->ab', cov, w, w)
        var = fnp.maximum(fnp.diag(cp), fnp.asarray(1e-15, dtype=fnp.float32))
        sigma = fnp.sqrt(var)
        alpha = mu / sigma
        phi = fnp.exp(-0.5 * alpha ** 2) / fnp.asarray(fnp.sqrt(2 * fnp.pi), dtype=fnp.float32)
        gain = _cdf(alpha)
        hs = hermites(alpha, max(degree, 8))
        rs = raw_moments(mu, var, sigma, phi, gain)
        if not final_pair_only:
            _review_prime_native_derivative_cache(sigma, phi, gain, hs, rs)
            gaussian_cov = _audit_covariance_readonly(cp, sigma, phi, gain, hs, degree)
        gram = None
        third_gram = cp
        third_gd = fnp.diag(third_gram)
        third_linear = third_a @ w
        skew_pre = 3 * third_gd * third_linear
        age_third_cup = skew_pre
        age_third_path, age_third_repair = ([], [])
        final_pair_sums = None
        if final_pair_only:
            pair_pre = None
            if readout_coefficients is not None and layer > 0:
                cup_row = third_gd * fnp.sum(third_linear) + 2 * third_linear * fnp.sum(third_gram, axis=1)
                cup_col = third_linear * fnp.sum(third_gd) + 2 * (third_linear @ third_gram)
                final_pair_sums = (cup_row, cup_col)
        else:
            pair_pre = third_gd[:, None] * third_linear[None, :] + 2 * third_linear[:, None] * third_gram
        transported_banks, pre_banks = ([], [])
        if banks:
            cp_wt = wt[:, cp_active_rows]
            newest_aw = cp_wt[:, cp_birth_rows] * fnp.sum(banks[-1][0], axis=0)[None, :]
            if len(banks) > 1:
                older_width = sum((a.shape[1] for a, b in banks[:-1]))
                transport_right = fnp.concatenate([a for a, b in banks[:-1]] + [b for a, b in banks], axis=1)
                transported_both = _adjusted_pooled_cp_product(cp_wt, transport_right)
                all_aw = fnp.concatenate([transported_both[:, :older_width], newest_aw], axis=1)
                all_bw = transported_both[:, older_width:]
            else:
                all_aw = newest_aw
                all_bw = _adjusted_pooled_cp_product(cp_wt, banks[0][1])
            all_ab = all_aw * all_bw
            all_skew_terms = all_ab * all_bw
            skew_pre = skew_pre + fnp.sum(all_skew_terms, axis=1)
            if final_pair_only:
                if readout_coefficients is not None and layer > 0:
                    cp_row, cp_col = _adjusted_cp_pair_sums(all_aw, all_bw, all_ab)
                    final_pair_sums = (final_pair_sums[0] + cp_row, final_pair_sums[1] + cp_col)
            else:
                fnp.add(pair_pre, (_adjusted_pooled_cp_product(all_bw * all_bw, all_aw.T) + _adjusted_pooled_cp_product(2 * all_ab, all_bw.T)) / 3, out=pair_pre)
            if not final_pair_only:
                all_post_a = all_aw * gain[:, None]
            if not final_pair_only:
                all_post_b = all_bw * gain[:, None]
            _derivative_cache['efficiency_full_cp_pre'] = (all_aw, all_bw)
            lo = 0
            for bank_index in range(len(banks)):
                hi = lo + banks[bank_index][0].shape[1]
                aw, bw = (all_aw[:, lo:hi], all_bw[:, lo:hi])
                age_third_path.append(fnp.sum(all_skew_terms[:, lo:hi], axis=1))
                if not final_pair_only:
                    pre_banks.append((aw, bw))
                if not final_pair_only:
                    transported_banks.append((all_post_a[:, lo:hi], all_post_b[:, lo:hi]))
                lo = hi
        fourth_gram = cp
        fourth_gd = fnp.diag(fourth_gram)
        transported_b = fnp.asarray(fourth_scalar, dtype=fnp.float32) * cp + fnp.einsum('i,ia,ib->ab', fourth_diagonal, w, w)
        bd = fnp.diag(transported_b)
        kurt_pre = 6 * fourth_gd * bd
        age_fourth_cup = kurt_pre
        age_fourth_star, age_fourth_path = ([], [])
        k31 = 3 * (fourth_gd[:, None] * transported_b + bd[:, None] * fourth_gram)
        k22 = fourth_gd[:, None] * bd[None, :] + bd[:, None] * fourth_gd[None, :] + 4 * fourth_gram * transported_b
        transported_fourth_paths = []
        transported_fourth_pairs = []
        transported_direct = []
        if final_pair_only:
            corrected = [rs[0], rs[1] + skew_pre * derivative(1, 3, sigma, phi, gain, hs, rs) / 6 + kurt_pre * derivative(1, 4, sigma, phi, gain, hs, rs) / 24]
        else:
            corrected = _review_corrected_moment_block(sigma, phi, gain, hs, rs, skew_pre, kurt_pre)
        mean = corrected[1]
        if layer == len(weights) - 1 and (not return_state):
            if readout_coefficients is not None and layer > 0:
                base, readout_context = _readout_features(cp, pair_pre, alpha, sigma, phi, gain, skew_pre, kurt_pre, include_higher=True, pair_sums=final_pair_sums, return_context=True)
                readout = _age_readout_features(base, alpha, sigma, phi, age_third_cup, age_third_path, age_third_repair, age_fourth_cup, age_fourth_star, age_fourth_path, context=readout_context)
                correction = correction @ w * gain + readout @ readout_coefficients[layer]
            rows.append(mean + correction)
            continue
        postvar = fnp.maximum(corrected[2] - mean * mean, fnp.asarray(1e-15, dtype=fnp.float32))
        h2 = phi / sigma
        h3 = -alpha * phi / var
        if correlation_order >= 0:
            d1 = [rs[1]] + [derivative(1, k, sigma, phi, gain, hs, rs) for k in range(1, max(1, correlation_order) + 4)]
            d2 = [rs[2] - rs[1] ** 2] + [_review_centered_square(k, sigma, phi, gain, hs, rs) for k in range(1, len(d1))]
        cov_third = pair_pre * h2[:, None] * gain[None, :] / 2
        cov_third = cov_third + cov_third.T
        delta_gain = skew_pre * derivative(1, 4, sigma, phi, gain, hs, rs) / 6
        cov_third = cov_third + cp * (delta_gain[:, None] * gain[None, :] + gain[:, None] * delta_gain[None, :])
        cov_delta = cov_third.copy()
        if fourth_correlation_order >= 0:
            highest = max(1, fourth_correlation_order) + 4
            fd1 = [rs[1]] + [derivative(1, k, sigma, phi, gain, hs, rs) for k in range(1, highest + 1)]
            fd2 = [rs[2] - rs[1] ** 2] + [derivative(2, k, sigma, phi, gain, hs, rs) - 2 * rs[1] * fd1[k] for k in range(1, highest + 1)]
        if fourth_correlation_order >= 0:
            cov_fourth = correlated_fourth_response(cp, kurt_pre, k31, k22, fd1, fd1, fourth_correlation_order)
        else:
            fourth_gain = kurt_pre * derivative(1, 5, sigma, phi, gain, hs, rs) / 24
            cov_fourth = cp * (fourth_gain[:, None] * gain[None, :] + gain[:, None] * fourth_gain[None, :])
            delta = k31 * h3[:, None] * gain[None, :] / 6
            cov_fourth = cov_fourth + (delta + delta.T)
            cov_fourth = cov_fourth + k22 * h2[:, None] * h2[None, :] / 4
        cov_delta = cov_delta + cov_fourth
        covariance_second = None
        cov = gaussian_cov + cov_delta
        fnp.fill_diagonal(cov, postvar)
        cov = flops.as_symmetric(0.5 * (cov + cov.T), symmetry=(0, 1))
        anew, bnew, pair_gaussian = gaussian_third_factors(cp, sigma, phi, gain, hs, rs, degree=min(degree, 6))
        g1 = 2 * rs[1] * (1 - gain)
        g2 = 2 * gain - 2 * rs[1] * h2
        g3 = 2 * h2 - 2 * rs[1] * h3
        if correlation_order >= 0:
            pair_delta = correlated_third_response(cp, skew_pre, pair_pre, d2, d1, correlation_order)
            pair_delta = pair_delta - 2 * (skew_pre * d1[3] / 6)[:, None] * gaussian_cov
            pair_delta = pair_delta - gain[:, None] ** 2 * gain[None, :] * pair_pre
        else:
            pair_delta = (0.5 * g2 - gain ** 2)[:, None] * gain[None, :] * pair_pre
            pair_delta = pair_delta + 0.5 * g1[:, None] * h2[None, :] * pair_pre.T
        if fourth_correlation_order >= 0:
            pair_delta = pair_delta + correlated_fourth_response(cp, kurt_pre, k31, k22, fd2, fd1, fourth_correlation_order)
            pair_delta = pair_delta - 2 * (kurt_pre * fd1[4] / 24)[:, None] * gaussian_cov
        else:
            pair_delta = pair_delta + k31 * g3[:, None] * gain[None, :] / 6
            pair_delta = pair_delta + k22 * g2[:, None] * h2[None, :] / 4
            pair_delta = pair_delta + k31.T * g1[:, None] * h3[None, :] / 6
        fnp.fill_diagonal(pair_delta, 0)
        pair_target = pair_gaussian + pair_delta + gain[:, None] ** 2 * gain[None, :] * pair_pre
        k3post = corrected[3] - 3 * mean * corrected[2] + 2 * mean ** 3
        k3gauss = _review_gaussian_k3(rs)
        residual = k3post - k3gauss - gain ** 3 * skew_pre
        edge_left = gain[:, None] * cp + 0.5 * h2[:, None] * pair_pre
        edge_right = gain[:, None] * pair_pre.T + 0.5 * h2[:, None] * k22
        edge_right = edge_right + h2[:, None] * cp ** 2
        bnew = edge_left - (alpha / (2 * sigma))[None, :] * edge_right
        banks = (transported_banks + [(anew, bnew)])[-horizon:]
        fnp.fill_diagonal(pair_target, k3post)
        full_trace = fnp.sum(pair_target, axis=0)
        cp_trace = _adjusted_current_bank_trace(banks, all_post_a if transported_banks else None, all_post_b if transported_banks else None, n)
        remainder_trace = full_trace - cp_trace
        trace_covariance = fnp.trace(cov, dtype=fnp.float64)
        third_a = remainder_trace / fnp.asarray(trace_covariance, dtype=fnp.float32)
        for _ in range(3):
            third_a = (remainder_trace - 2 * cov @ third_a) / fnp.asarray(trace_covariance, dtype=fnp.float32)
        birth_degree = degree
        birth = fourth_birth_trace_power(cp, sigma, phi, gain, hs, rs, degree=birth_degree, offdiag=False, edge_degree=gaussian_trace_edge_degree)
        trace_diagonal = fnp.diag(birth) + _adjusted_inherited_trace_diagonal(fourth_gram, transported_b, gain)
        feedback_cov_delta = cov_delta if covariance_second is None else cov_delta - covariance_second
        feedback_diag = square_cumulant_feedback(cp, gaussian_cov, gain, rs, corrected, sigma, phi, hs, skew_pre, pair_pre, kurt_pre, k31, k22, feedback_cov_delta, include_fourth=include_fourth_feedback, consistent_third_only=consistent_third_only, third_covariance=cov_third)
        if fourth_correlation_order >= 0:
            baseline_third = square_cumulant_feedback(cp, gaussian_cov, gain, rs, corrected, sigma, phi, hs, skew_pre, pair_pre, kurt_pre, k31, k22, cov_delta, include_fourth=False, consistent_third_only=True, third_covariance=cov_third)
            feedback_diag = baseline_third + correlated_fourth_square(cp, gaussian_cov, pair_gaussian, gain, rs, sigma, phi, hs, kurt_pre, k31, k22, fourth_correlation_order, fd1, fd2, cov_fourth)
        trace_diagonal = trace_diagonal + feedback_diag
        direction_u = fnp.linalg.svd(cov, full_matrices=False, k=1, hermitian=True)[0][:, 0]
        direction_target = _adjusted_target_trace_quadratic(cp, transported_b, gain, rs, sigma, phi, hs, third_gram, third_linear, pre_banks, gaussian_cov, pair_gaussian, skew_pre, pair_pre, cov_third, feedback_diag, direction_u)
        fourth_scalar, fourth_diagonal = _adjusted_fit_directional_gauge(trace_diagonal, cov, direction_u, direction_target)
        if readout_coefficients is not None and layer > 0:
            base, readout_context = _readout_features(cp, pair_pre, alpha, sigma, phi, gain, skew_pre, kurt_pre, include_higher=True, pair_sums=final_pair_sums, return_context=True)
            readout = _age_readout_features(base, alpha, sigma, phi, age_third_cup, age_third_path, age_third_repair, age_fourth_cup, age_fourth_star, age_fourth_path, context=readout_context)
            correction = correction @ w * gain + readout @ readout_coefficients[layer]
        banks, cp_active_rows, compact_trace, cp_birth_rows = _adjusted_compact_activation_cp(banks, gain, activation_cp_threshold, activation_cp_upper, all_post_a if transported_banks else None, all_post_b if transported_banks else None)
        if compact_trace is not None:
            discarded_trace = cp_trace - compact_trace
            trace_change = discarded_trace / fnp.asarray(trace_covariance, dtype=fnp.float32)
            for _ in range(3):
                trace_change = (discarded_trace - 2 * cov @ trace_change) / fnp.asarray(trace_covariance, dtype=fnp.float32)
            third_a = third_a + trace_change
        rows.append(mean + correction)
        if return_state:
            state_history.append({'mean_pre': mu, 'sigma': sigma, 'alpha': alpha, 'gain': gain, 'phi': phi, 'skew': skew_pre, 'kurt': kurt_pre})
    if return_state:
        state = {'mean_pre': mu, 'sigma': sigma, 'alpha': alpha, 'gain': gain, 'phi': phi, 'skew': skew_pre, 'kurt': kurt_pre, 'history': state_history}
        return (fnp.stack(rows, axis=0), diagnostics, state)
    return (fnp.stack(rows, axis=0), diagnostics)

class Estimator(BaseEstimator):
    path_horizon = 16

    def setup(self, ctx: SetupContext) -> None:
        self.readout_coefficients = fnp.asarray(READOUT_COEFFICIENTS, dtype=fnp.float64)
        self.cp_engine = AdjustedRaggedRightPlanCP(depth=5, capacity=min(1024, max(32, (ctx.width + 31) // 32 * 32)), batch=8, min_size=128)
        self.cp_engine.warm_common_plans()
        # Initialize every fixed scratch buffer during setup. No model data is used.
        for workspace in self.cp_engine.buffers.values():
            fnp.copyto(workspace, 0)

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        global _adjusted_pooled_cp_engine
        _adjusted_pooled_cp_engine = self.cp_engine
        rows, _ = _predict(mlp.weights, **PREDICTION_CONFIG, sketch_seed=mlp.seed, readout_coefficients=self.readout_coefficients)
        self.cp_engine.reset_prediction()
        return rows.astype(fnp.float32)
EDGE_STRENGTH = 0.3

def _efficiency_covariance_basis(covariance, rank, iterations=12, sketch_seed=2026):
    n = len(covariance)
    rank = min(rank, n)
    if rank == n:
        return fnp.eye(n, dtype=covariance.dtype)
    sketch = fnp.random.default_rng(sketch_seed).standard_normal((n, min(n, rank + 8))).astype(covariance.dtype)
    basis = fnp.linalg.qr(sketch, mode='reduced')[0]
    for _ in range(iterations):
        basis = fnp.linalg.qr(covariance @ basis, mode='reduced')[0]
    small = basis.T @ covariance @ basis
    values, rotation = fnp.linalg.eigh((small + small.T) / 2)
    chosen = fnp.argsort(values)[-rank:]
    return (basis @ rotation[:, chosen]).astype(covariance.dtype)

def _efficiency_project_b_slots(a, b, basis):
    coordinate = basis.T @ b
    rank = basis.shape[1]
    _audit_local_transpose_coordinate = coordinate.T
    product = _audit_local_transpose_coordinate[:, :, None] * _audit_local_transpose_coordinate[:, None, :]
    coefficient = (a @ product.reshape(a.shape[1], rank * rank)).reshape(a.shape[0], rank, rank)
    diagonal = list(range(rank))
    ii = [i for i in range(rank) for j in range(i + 1, rank)]
    jj = [j for i in range(rank) for j in range(i + 1, rank)]
    cross = (coefficient[:, ii, jj] + coefficient[:, jj, ii]) / 4
    first = fnp.concatenate([coefficient[:, diagonal, diagonal], fnp.stack([cross, -cross], axis=-1).reshape(a.shape[0], -1)], axis=1)
    plus = basis[:, ii] + basis[:, jj]
    minus = basis[:, ii] - basis[:, jj]
    second = fnp.concatenate([basis, fnp.stack([plus, minus], axis=-1).reshape(a.shape[0], -1)], axis=1)
    return (first, second)

def _audit_covariance_readonly(cov, sigma, phi, prob, hs, degree):
    if degree not in _covariance_cache:
        _covariance_cache[degree] = _covariance_hermite_uncached(cov, sigma, phi, prob, hs, degree)
    return _covariance_cache[degree]

def _adjusted_inherited_trace_diagonal(gram, transported_b, gain):
    gain2 = gain * gain
    cg = fnp.diag(gram)
    cb = fnp.diag(transported_b)
    tg = fnp.sum(gain2 * cg, dtype=fnp.float64).astype(fnp.float32)
    tb = fnp.sum(gain2 * cb, dtype=fnp.float64).astype(fnp.float32)
    cross = fnp.sum(gram * gain2[None, :] * transported_b.T, axis=1, dtype=fnp.float64).astype(fnp.float32)
    return gain2 * (cg * tb + cb * tg + 4 * cross)

def _adjusted_fit_scalar_trace(diag_target, covariance):
    trace = fnp.sum(fnp.diag(covariance), dtype=fnp.float64)
    norm2 = fnp.sum(covariance * covariance, dtype=fnp.float64)
    target = fnp.sum(diag_target, dtype=fnp.float64)
    return (target / fnp.maximum(2 * (trace * trace + 2 * norm2), 1e-30)).astype(fnp.float32)

def _adjusted_fit_diagonal_trace(diag_target, covariance):
    diagonal = fnp.diag(covariance)
    trace = fnp.sum(diagonal, dtype=fnp.float64).astype(fnp.float32)
    row2 = fnp.sum(covariance * covariance, axis=1, dtype=fnp.float64).astype(fnp.float32)
    denominator = fnp.maximum(trace + 4 * diagonal, 1e-15)
    coherent = diagonal * trace + 2 * row2
    numerator = fnp.sum(diag_target / denominator, dtype=fnp.float64)
    metric = fnp.sum(coherent / denominator, dtype=fnp.float64)
    theta = (numerator / fnp.maximum(2 * metric, 1e-30)).astype(fnp.float32)
    correction = (diag_target - 2 * theta * coherent) / denominator
    return (theta, correction)

def _adjusted_gaussian_trace_quadratic(cp, u):
    cp2, p, h2, g1, g2, k31, trace_diagonal, variance = _derivative_cache['adjusted_gaussian_directional']
    x = p * u
    cx = cp @ x
    central = g2 - 2 * p * p
    value = fnp.sum(central * cx * cx, dtype=fnp.float64)
    value = value - 2 * fnp.sum(x * variance * central * cx, dtype=fnp.float64)
    cg1 = cp @ g1
    value = value + 2 * fnp.sum(u * h2 * (cg1 - variance * g1) * cx, dtype=fnp.float64)
    value = value - 2 * fnp.sum(u * h2 * (cp2 @ (x * g1)), dtype=fnp.float64)
    value = value + 2 * fnp.sum(u * (k31 @ u), dtype=fnp.float64)
    generated_diagonal = p * p * (cp2 @ central - 2 * variance * variance * central)
    generated_diagonal = generated_diagonal + 2 * variance * h2 * p * (cg1 - 2 * variance * g1)
    generated_diagonal = generated_diagonal + 2 * fnp.diag(k31)
    return value + fnp.sum(u * u * (trace_diagonal - generated_diagonal), dtype=fnp.float64)

def _adjusted_inherited_trace_quadratic(gram, transported_b, gain, u):
    gain2 = gain * gain
    tg = fnp.sum(gain2 * fnp.diag(gram), dtype=fnp.float64)
    tb = fnp.sum(gain2 * fnp.diag(transported_b), dtype=fnp.float64)
    v = gain * u
    cv = gram @ v
    bv = transported_b @ v
    return tb * fnp.sum(v * cv, dtype=fnp.float64) + tg * fnp.sum(v * bv, dtype=fnp.float64) + 4 * fnp.sum(cv * gain2 * bv, dtype=fnp.float64)

def _adjusted_third_trace_quadratic(gram, third_linear, pre_banks, gain, rs, sigma, phi, hs, cp, gaussian_cov, pair_gaussian, skew, pair, covariance_third, u):
    mu = rs[1]
    g1 = 2 * mu * (1 - gain)
    v = gain * u
    gram_g1 = gram @ g1
    value = fnp.dot(third_linear, g1) * fnp.sum(v * (gram @ v), dtype=fnp.float64)
    value = value + 2 * fnp.dot(v, third_linear) * fnp.sum(v * gram_g1, dtype=fnp.float64)
    if pre_banks:
        if 'efficiency_full_cp_pre' in _derivative_cache:
            a, b = _derivative_cache['efficiency_full_cp_pre']
        else:
            a = fnp.concatenate([af for af, bf in pre_banks], axis=1)
            b = fnp.concatenate([bf for af, bf in pre_banks], axis=1)
        av, bv = (a.T @ v, b.T @ v)
        ag, bg = (a.T @ g1, b.T @ g1)
        value = value + fnp.sum(ag * bv * bv + 2 * av * bv * bg, dtype=fnp.float64) / 3
    value = value - 2 * fnp.sum(v * g1 * (pair @ v), dtype=fnp.float64)
    d1 = [rs[1]] + [derivative(1, k, sigma, phi, gain, hs, rs) for k in range(1, 5)]
    cube = [_review_gaussian_k3(rs)]
    _, centered_directional = _review_centered_cube_family(sigma, phi, gain, hs, rs, upto=4)
    cube = cube + [centered_directional[k - 1] for k in range(1, 5)]
    delta = correlated_third_response(cp, skew, pair, cube, d1, 1)
    delta_mean = skew * d1[3] / 6
    delta_variance = skew * (derivative(2, 3, sigma, phi, gain, hs, rs) - 2 * mu * d1[3]) / 6
    variance = rs[2] - mu ** 2
    delta = delta - 3 * delta_mean[:, None] * pair_gaussian
    delta = delta - 3 * delta_variance[:, None] * gaussian_cov
    delta = delta - 3 * variance[:, None] * covariance_third
    value = value + 2 * fnp.sum(u * (delta @ u), dtype=fnp.float64)
    diagonal = gain * gain * (pair @ g1 - 2 * fnp.diag(pair) * g1) + 2 * fnp.diag(delta)
    return value - fnp.sum(u * u * diagonal, dtype=fnp.float64)

def _adjusted_target_trace_quadratic(cp, transported_b, gain, rs, sigma, phi, hs, third_gram, third_linear, pre_banks, gaussian_cov, pair_gaussian, skew, pair, covariance_third, feedback_diag, u):
    value = _adjusted_gaussian_trace_quadratic(cp, u)
    value = value + _adjusted_inherited_trace_quadratic(cp, transported_b, gain, u)
    value = value + _adjusted_third_trace_quadratic(third_gram, third_linear, pre_banks, gain, rs, sigma, phi, hs, cp, gaussian_cov, pair_gaussian, skew, pair, covariance_third, u)
    return value + fnp.sum(u * u * feedback_diag, dtype=fnp.float64)

def _adjusted_fit_directional_gauge(diag_target, covariance, u, target_quadratic):
    diagonal = fnp.diag(covariance)
    trace = fnp.sum(diagonal, dtype=fnp.float64).astype(fnp.float32)
    row2 = fnp.sum(covariance * covariance, axis=1, dtype=fnp.float64).astype(fnp.float32)
    denominator = fnp.maximum(trace + 4 * diagonal, 1e-15)
    metric_diagonal = 2 * diagonal * trace + 4 * row2
    total_denominator = 1 + fnp.sum(diagonal / denominator)
    total0 = fnp.sum(diag_target / denominator) / total_denominator
    total1 = fnp.sum(metric_diagonal / denominator) / total_denominator
    d0 = (diag_target - diagonal * total0) / denominator
    d1 = (-metric_diagonal + diagonal * total1) / denominator
    cu = covariance @ u
    uc = fnp.sum(u * cu, dtype=fnp.float64)
    q0 = total0 * uc + trace * fnp.sum(d0 * u * u, dtype=fnp.float64) + 4 * fnp.sum(d0 * u * cu, dtype=fnp.float64)
    q1 = (2 * trace - total1) * uc + 4 * fnp.sum(cu * cu, dtype=fnp.float64)
    q1 = q1 + trace * fnp.sum(d1 * u * u, dtype=fnp.float64) + 4 * fnp.sum(d1 * u * cu, dtype=fnp.float64)
    denominator_q = fnp.where(fnp.abs(q1) >= 1e-30, q1, fnp.where(q1 < 0, -1e-30, 1e-30))
    theta = ((target_quadratic - q0) / denominator_q).astype(fnp.float32)
    return (theta, d0 + theta * d1)

def _adjusted_fit_directional_rankone(diag_target, covariance, u, target_quadratic):
    theta, d0 = _adjusted_fit_diagonal_trace(diag_target, covariance)
    diagonal = fnp.diag(covariance)
    trace = fnp.sum(diagonal, dtype=fnp.float64).astype(fnp.float32)
    denominator = fnp.maximum(trace + 4 * diagonal, 1e-15)
    cu = covariance @ u
    uc = fnp.sum(u * cu, dtype=fnp.float64)
    norm = fnp.sum(u * u, dtype=fnp.float64)
    rank_diagonal = trace * u * u + norm.astype(fnp.float32) * diagonal + 4 * u * cu
    rank_total = fnp.sum(rank_diagonal / denominator) / (1 + fnp.sum(diagonal / denominator))
    dr = (-rank_diagonal + diagonal * rank_total) / denominator
    q0 = 2 * theta * (trace * uc + 2 * fnp.sum(cu * cu, dtype=fnp.float64))
    q0 = q0 + fnp.sum(d0, dtype=fnp.float64) * uc + trace * fnp.sum(d0 * u * u, dtype=fnp.float64) + 4 * fnp.sum(d0 * u * cu, dtype=fnp.float64)
    qr = (norm - rank_total) * uc + trace * norm * norm + 4 * norm * uc
    qr = qr + trace * fnp.sum(dr * u * u, dtype=fnp.float64) + 4 * fnp.sum(dr * u * cu, dtype=fnp.float64)
    denominator_q = fnp.where(fnp.abs(qr) >= 1e-30, qr, fnp.where(qr < 0, -1e-30, 1e-30))
    strength = ((target_quadratic - q0) / denominator_q).astype(fnp.float32)
    return (theta, d0 + strength * dr, strength)

def _adjusted_cp_pair_sums(a, b, ab):
    sum_a = fnp.sum(a, axis=0)
    sum_b = fnp.sum(b, axis=0)
    b2 = b * b
    sum_b2 = fnp.sum(b2, axis=0)
    sum_ab = fnp.sum(ab, axis=0)
    row = (b2 @ sum_a + 2 * ab @ sum_b) / 3
    col = (a @ sum_b2 + 2 * b @ sum_ab) / 3
    return (row, col)

def _adjusted_packed_trace(a, b):
    return (a @ fnp.sum(b * b, axis=0) + 2 * b @ fnp.sum(a * b, axis=0)) / 3

def _adjusted_current_bank_trace(banks, old_a, old_b, total_size):
    result = fnp.zeros(total_size, dtype=fnp.float32)
    if len(banks) > 1:
        width = sum((a.shape[1] for a, b in banks[:-1]))
        start = old_a.shape[1] - width
        result = _adjusted_packed_trace(old_a[:, start:], old_b[:, start:])
    a, b = banks[-1]
    diagonal = fnp.diag(a)
    return result + (diagonal * fnp.sum(b * b, axis=0) + 2 * b @ (diagonal * fnp.diag(b))) / 3

def _adjusted_compact_activation_cp(banks, gain, threshold, upper, old_a, old_b):
    active = fnp.nonzero(gain >= fnp.asarray(threshold, dtype=gain.dtype))[0]
    centers = fnp.nonzero(gain[active] <= fnp.asarray(upper, dtype=gain.dtype))[0]
    if len(active) == len(gain) and len(centers) == len(gain):
        return (banks, active, None, centers)
    compact = []
    trace = fnp.zeros(len(active), dtype=gain.dtype)
    if len(banks) > 1:
        width = sum((a.shape[1] for a, b in banks[:-1]))
        start = old_a.shape[1] - width
        a = old_a[active][:, start:]
        b = old_b[active][:, start:]
        trace = _adjusted_packed_trace(a, b)
        lo = 0
        for previous_a, previous_b in banks[:-1]:
            hi = lo + previous_a.shape[1]
            compact.append((a[:, lo:hi], b[:, lo:hi]))
            lo = hi
    a, b = banks[-1]
    a = fnp.diag(fnp.diag(a)[active])[:, centers]
    b = b[active][:, active[centers]]
    compact.append((a, b))
    trace = trace + _adjusted_packed_trace(a, b)
    if len(active):
        positions = fnp.maximum(fnp.cumsum((gain >= threshold).astype(fnp.int64)) - 1, 0)
        full_trace = fnp.where(gain >= threshold, trace[positions], fnp.asarray(0, dtype=gain.dtype))
    else:
        full_trace = fnp.zeros(len(gain), dtype=gain.dtype)
    return (compact, active, full_trace, centers)

class PooledWinograd:

    def __init__(self, depth=5, min_block=16):
        self.depth = depth
        self.min_block = min_block
        self.buffers = {}
        self.owned_buffers = set()
        self.view_cache = {}

    def buffer(self, role, shape, dtype):
        key = (role, tuple(shape), str(dtype))
        if key not in self.buffers:
            result = fnp.empty(shape, dtype=dtype)
            fnp.copyto(result, 0)
            self.buffers[key] = result
            self.owned_buffers.add(id(result))
        return self.buffers[key]

    def slots(self, value):
        key = ('slots', id(value))
        if key not in self.view_cache:
            self.view_cache[key] = tuple((value[i] for i in range(7)))
        return self.view_cache[key]

    def quadrants(self, value):
        key = ('quadrants', id(value))
        if id(value) in self.owned_buffers and key in self.view_cache:
            return self.view_cache[key]
        half_rows, half_columns = (value.shape[-2] // 2, value.shape[-1] // 2)
        result = (value[..., :half_rows, :half_columns], value[..., :half_rows, half_columns:], value[..., half_rows:, :half_columns], value[..., half_rows:, half_columns:])
        if id(value) in self.owned_buffers:
            self.view_cache[key] = result
        return result

    def multiply(self, left, right, out=None):
        a, b = (left, right)
        m, inner, n = (a.shape[-2], a.shape[-1], b.shape[-1])
        if b.shape[-2] != inner:
            raise ValueError('Incompatible matrix dimensions.')
        if a.dtype != b.dtype:
            raise ValueError('Use one explicit common arithmetic dtype.')
        ashape, bshape = (a.shape[:-2], b.shape[:-2])
        count = max(len(ashape), len(bshape))
        ashape = (1,) * (count - len(ashape)) + ashape
        bshape = (1,) * (count - len(bshape)) + bshape
        batch = []
        for x, y in zip(ashape, bshape):
            if x != y and x != 1 and (y != 1):
                raise ValueError('Incompatible leading batch dimensions.')
            batch.append(max(x, y))
        if out is None:
            out = fnp.empty(tuple(batch) + (m, n), dtype=a.dtype)
        self._multiply(a, b, out, self.depth)
        return out

    def _multiply(self, a, b, out, depth):
        m, inner, n = (a.shape[-2], a.shape[-1], b.shape[-1])
        if depth <= 0 or min(m, inner, n) < 2 * self.min_block or any((x % 2 for x in (m, inner, n))):
            fnp.matmul(a, b, out=out)
            return
        h, q, v = (m // 2, inner // 2, n // 2)
        a11, a12, a21, a22 = self.quadrants(a)
        b11, b12, b21, b22 = self.quadrants(b)
        ac = self.buffer(('A', depth), (7,) + a.shape[:-2] + (h, q), a.dtype)
        bc = self.buffer(('B', depth), (7,) + b.shape[:-2] + (q, v), b.dtype)
        products = self.buffer(('P', depth), (7,) + out.shape[:-2] + (h, v), out.dtype)
        av, bv, pv = (self.slots(ac), self.slots(bc), self.slots(products))
        fnp.copyto(av[0], a11)
        fnp.copyto(av[1], a12)
        fnp.add(a21, a22, out=av[4])
        fnp.subtract(av[4], a11, out=av[5])
        fnp.subtract(a11, a21, out=av[6])
        fnp.subtract(a12, av[5], out=av[2])
        fnp.copyto(av[3], a22)
        fnp.copyto(bv[0], b11)
        fnp.copyto(bv[1], b21)
        fnp.subtract(b12, b11, out=bv[4])
        fnp.subtract(b22, bv[4], out=bv[5])
        fnp.subtract(b22, b12, out=bv[6])
        fnp.subtract(bv[5], b21, out=bv[3])
        fnp.copyto(bv[2], b22)
        self._multiply(ac, bc, products, depth - 1)
        c11, c12, c21, c22 = self.quadrants(out)
        fnp.add(pv[0], pv[1], out=c11)
        fnp.add(pv[0], pv[5], out=pv[0])
        fnp.add(pv[0], pv[6], out=pv[6])
        fnp.add(pv[0], pv[4], out=pv[0])
        fnp.add(pv[0], pv[2], out=c12)
        fnp.subtract(pv[6], pv[3], out=c21)
        fnp.add(pv[6], pv[4], out=c22)

class AdjustedPooledCPTiled(PooledWinograd):

    def __init__(self, depth=3, capacity=1024, batch=4, min_size=128, min_block=16):
        super().__init__(depth=depth, min_block=min_block)
        if capacity > 1024 or capacity % 32 or batch < 1 or (batch > 8):
            raise ValueError('Use capacity<=1024 in multiples of32 and batch<=8.')
        self.capacity = capacity
        self.batch = batch
        self.min_size = min_size
        self.capacity_bytes = 0

    def buffer(self, role, shape, dtype):
        shape = tuple(shape)
        key = (role, len(shape), str(dtype))
        if key not in self.buffers:
            level = self.depth - role[1] + 1 if isinstance(role, tuple) else 0
            side = self.capacity // 2 ** level
            capshape = shape[:-3] + (self.batch, side, side)
            value = fnp.empty(capshape, dtype=dtype)
            fnp.copyto(value, 0)
            self.buffers[key] = value
            self.owned_buffers.add(id(value))
            self.capacity_bytes += value.size * value.dtype.itemsize
        storage = self.buffers[key]
        if any((a > b for a, b in zip(shape, storage.shape))):
            raise ValueError('Requested view exceeds fixed pool capacity.')
        viewkey = ('capacity_view', key, shape)
        if viewkey not in self.view_cache:
            value = storage[tuple((slice(0, n) for n in shape))]
            self.view_cache[viewkey] = value
            self.owned_buffers.add(id(value))
        return self.view_cache[viewkey]

    def tile_slots(self, value):
        key = ('tile_slots', id(value))
        if key not in self.view_cache:
            self.view_cache[key] = tuple((value[i] for i in range(value.shape[0])))
        return self.view_cache[key]

    def cropped(self, value, rows, columns):
        key = ('io_crop', id(value), rows, columns)
        if key not in self.view_cache:
            self.view_cache[key] = value[:rows, :columns]
        return self.view_cache[key]

    def product(self, left, right):
        if left.ndim != 2 or right.ndim != 2:
            return fnp.matmul(left, right)
        m, k = left.shape
        n = right.shape[1]
        if right.shape[0] != k:
            raise ValueError('Incompatible matrix dimensions.')
        if self.depth <= 0 or min(m, k, n) < self.min_size or left.dtype != right.dtype:
            return fnp.matmul(left, right)
        output = fnp.empty((m, n), dtype=left.dtype)
        if k <= self.capacity:
            self._wide(left, right, output)
        else:
            self._long_inner(left, right, output)
        return output

    def _views(self, count, rows, inner, columns, dtype):
        a = self.buffer('ioA', (count, rows, inner), dtype)
        b = self.buffer('ioB', (count, inner, columns), dtype)
        c = self.buffer('ioC', (count, rows, columns), dtype)
        return (a, b, c, self.tile_slots(a), self.tile_slots(b), self.tile_slots(c))

    def _wide(self, left, right, output):
        m, k = left.shape
        n = right.shape[1]
        cap = self.capacity
        kp = (k + 31) // 32 * 32
        column_tiles = (n + cap - 1) // cap
        for row0 in range(0, m, cap):
            rows = min(cap, m - row0)
            rp = (rows + 31) // 32 * 32
            for tile0 in range(0, column_tiles, self.batch):
                count = min(self.batch, column_tiles - tile0)
                columns = min(cap, n - tile0 * cap)
                cp = (columns + 31) // 32 * 32
                a, b, c, av, bv, cv = self._views(count, rp, kp, cp, left.dtype)
                fnp.copyto(a, 0)
                fnp.copyto(b, 0)
                source = left[row0:row0 + rows, :]
                fnp.copyto(a[:, :rows, :k], source[None, :, :])
                for slot in range(count):
                    col0 = (tile0 + slot) * cap
                    cols = min(cap, n - col0)
                    fnp.copyto(self.cropped(bv[slot], k, cols), right[:, col0:col0 + cols])
                self._multiply(a, b, c, self.depth)
                for slot in range(count):
                    col0 = (tile0 + slot) * cap
                    cols = min(cap, n - col0)
                    fnp.copyto(output[row0:row0 + rows, col0:col0 + cols], self.cropped(cv[slot], rows, cols))

    def _long_inner(self, left, right, output):
        m, k = left.shape
        n = right.shape[1]
        cap = self.capacity
        inner_tiles = (k + cap - 1) // cap
        for row0 in range(0, m, cap):
            rows = min(cap, m - row0)
            rp = (rows + 31) // 32 * 32
            for col0 in range(0, n, cap):
                cols = min(cap, n - col0)
                cp = (cols + 31) // 32 * 32
                destination = output[row0:row0 + rows, col0:col0 + cols]
                for tile0 in range(0, inner_tiles, self.batch):
                    count = min(self.batch, inner_tiles - tile0)
                    inner = min(cap, k - tile0 * cap)
                    kp = (inner + 31) // 32 * 32
                    a, b, c, av, bv, cv = self._views(count, rp, kp, cp, left.dtype)
                    fnp.copyto(a, 0)
                    fnp.copyto(b, 0)
                    for slot in range(count):
                        k0 = (tile0 + slot) * cap
                        span = min(cap, k - k0)
                        fnp.copyto(self.cropped(av[slot], rows, span), left[row0:row0 + rows, k0:k0 + span])
                        fnp.copyto(self.cropped(bv[slot], span, cols), right[k0:k0 + span, col0:col0 + cols])
                    self._multiply(a, b, c, self.depth)
                    partial = fnp.sum(c, axis=0)
                    partial = partial[:rows, :cols]
                    if tile0 == 0:
                        fnp.copyto(destination, partial)
                    else:
                        fnp.add(destination, partial, out=destination)

class AdjustedPooledCPPrepared(AdjustedPooledCPTiled):

    def __init__(self, depth=3, capacity=1024, batch=4, min_size=128, min_block=16):
        super().__init__(depth, capacity, batch, min_size, min_block)
        self.left_plans = {}
        self.right_plans = {}
        self.prepared_source = None
        self.prepared_key = None

    def buffer(self, role, shape, dtype):
        shape = tuple(shape)
        key = (role, len(shape), str(dtype))
        viewkey = ('capacity_view', key, shape)
        cached = self.view_cache.get(viewkey)
        if cached is not None:
            return cached
        storage = self.buffers.get(key)
        if storage is None:
            level = self.depth - role[1] + 1 if isinstance(role, tuple) else 0
            side = self.capacity // 2 ** level
            capshape = shape[:-3] + (self.batch, side, side)
            storage = fnp.empty(capshape, dtype=dtype)
            self.buffers[key] = storage
            self.owned_buffers.add(id(storage))
            self.capacity_bytes += storage.size * storage.dtype.itemsize
        if any((a > b for a, b in zip(shape, storage.shape))):
            raise ValueError('Requested view exceeds fixed pool capacity.')
        value = storage[tuple((slice(0, n) for n in shape))]
        self.view_cache[viewkey] = value
        self.owned_buffers.add(id(value))
        return value

    def left_plan(self, a, depth):
        key = (id(a), depth)
        cached = self.left_plans.get(key)
        if cached is not None:
            return cached
        m, inner = a.shape[-2:]
        if depth <= 0 or min(m, inner) < 2 * self.min_block or m % 2 or inner % 2:
            plan = (a, None, None)
        else:
            ac = self.buffer(('A', depth), (7,) + a.shape[:-2] + (m // 2, inner // 2), a.dtype)
            plan = (a, (self.quadrants(a), self.slots(ac)), self.left_plan(ac, depth - 1))
        self.left_plans[key] = plan
        return plan

    def prepare_left(self, plan):
        a, stage, child = plan
        if stage is None:
            return
        (a11, a12, a21, a22), av = stage
        fnp.copyto(av[0], a11)
        fnp.copyto(av[1], a12)
        fnp.add(a21, a22, out=av[4])
        fnp.subtract(av[4], a11, out=av[5])
        fnp.subtract(a11, a21, out=av[6])
        fnp.subtract(a12, av[5], out=av[2])
        fnp.copyto(av[3], a22)
        self.prepare_left(child)

    def right_plan(self, left_plan, b, out, depth):
        a = left_plan[0]
        key = (id(a), id(b), id(out), depth)
        cached = self.right_plans.get(key)
        if cached is not None:
            return cached
        m, inner, n = (a.shape[-2], a.shape[-1], b.shape[-1])
        if depth <= 0 or left_plan[1] is None or min(m, inner, n) < 2 * self.min_block or n % 2:
            plan = (a, b, out, None)
        else:
            bc = self.buffer(('B', depth), (7,) + b.shape[:-2] + (inner // 2, n // 2), b.dtype)
            products = self.buffer(('P', depth), (7,) + out.shape[:-2] + (m // 2, n // 2), out.dtype)
            stage = (self.quadrants(b), self.slots(bc), self.slots(products), self.quadrants(out), self.right_plan(left_plan[2], bc, products, depth - 1))
            plan = (a, b, out, stage)
        self.right_plans[key] = plan
        return plan

    def multiply_prepared(self, plan):
        a, b, out, stage = plan
        if stage is None:
            fnp.matmul(a, b, out=out)
            return
        (b11, b12, b21, b22), bv, pv, (c11, c12, c21, c22), child = stage
        fnp.copyto(bv[0], b11)
        fnp.copyto(bv[1], b21)
        fnp.subtract(b12, b11, out=bv[4])
        fnp.subtract(b22, bv[4], out=bv[5])
        fnp.subtract(b22, b12, out=bv[6])
        fnp.subtract(bv[5], b21, out=bv[3])
        fnp.copyto(bv[2], b22)
        self.multiply_prepared(child)
        fnp.add(pv[0], pv[1], out=c11)
        fnp.add(pv[0], pv[5], out=pv[0])
        fnp.add(pv[0], pv[6], out=pv[6])
        fnp.add(pv[0], pv[4], out=pv[0])
        fnp.add(pv[0], pv[2], out=c12)
        fnp.subtract(pv[6], pv[3], out=c21)
        fnp.add(pv[6], pv[4], out=c22)

    def _wide(self, left, right, output):
        m, k = left.shape
        n = right.shape[1]
        cap = self.capacity
        kp = (k + 31) // 32 * 32
        column_tiles = (n + cap - 1) // cap
        for row0 in range(0, m, cap):
            rows = min(cap, m - row0)
            rp = (rows + 31) // 32 * 32
            a = self.buffer('ioA', (1, rp, kp), left.dtype)
            lp = self.left_plan(a, self.depth)
            source_key = (row0, rows, k, rp, kp)
            if self.prepared_source is not left or self.prepared_key != source_key:
                if rows != rp or k != kp:
                    fnp.copyto(a, 0)
                fnp.copyto(a[0, :rows, :k], left[row0:row0 + rows, :])
                self.prepare_left(lp)
                self.prepared_source = left
                self.prepared_key = source_key
            for tile0 in range(0, column_tiles, self.batch):
                count = min(self.batch, column_tiles - tile0)
                columns = min(cap, n - tile0 * cap)
                cp = (columns + 31) // 32 * 32
                b = self.buffer('ioB', (count, kp, cp), right.dtype)
                c = self.buffer('ioC', (count, rp, cp), output.dtype)
                bv, cv = (self.tile_slots(b), self.tile_slots(c))
                last_cols = min(cap, n - (tile0 + count - 1) * cap)
                if k != kp or last_cols != cp:
                    fnp.copyto(b, 0)
                for slot in range(count):
                    col0 = (tile0 + slot) * cap
                    cols = min(cap, n - col0)
                    fnp.copyto(self.cropped(bv[slot], k, cols), right[:, col0:col0 + cols])
                self.multiply_prepared(self.right_plan(lp, b, c, self.depth))
                for slot in range(count):
                    col0 = (tile0 + slot) * cap
                    cols = min(cap, n - col0)
                    fnp.copyto(output[row0:row0 + rows, col0:col0 + cols], self.cropped(cv[slot], rows, cols))

    def _long_inner(self, left, right, output):
        self.prepared_source = None
        self.prepared_key = None
        super()._long_inner(left, right, output)

class AdjustedPooledCPPreparedLong(AdjustedPooledCPPrepared):

    def _long_inner(self, left, right, output):
        self.prepared_source = None
        self.prepared_key = None
        m, k = left.shape
        n = right.shape[1]
        cap = self.capacity
        inner_tiles = (k + cap - 1) // cap
        for row0 in range(0, m, cap):
            rows = min(cap, m - row0)
            rp = (rows + 31) // 32 * 32
            for col0 in range(0, n, cap):
                cols = min(cap, n - col0)
                cp = (cols + 31) // 32 * 32
                destination = output[row0:row0 + rows, col0:col0 + cols]
                for tile0 in range(0, inner_tiles, self.batch):
                    count = min(self.batch, inner_tiles - tile0)
                    inner = min(cap, k - tile0 * cap)
                    kp = (inner + 31) // 32 * 32
                    a, b, c, av, bv, cv = self._views(count, rp, kp, cp, left.dtype)
                    last_span = min(cap, k - (tile0 + count - 1) * cap)
                    if rows != rp or last_span != kp:
                        fnp.copyto(a, 0)
                    if cols != cp or last_span != kp:
                        fnp.copyto(b, 0)
                    for slot in range(count):
                        k0 = (tile0 + slot) * cap
                        span = min(cap, k - k0)
                        fnp.copyto(self.cropped(av[slot], rows, span), left[row0:row0 + rows, k0:k0 + span])
                        fnp.copyto(self.cropped(bv[slot], span, cols), right[k0:k0 + span, col0:col0 + cols])
                    lp = self.left_plan(a, self.depth)
                    self.prepare_left(lp)
                    self.multiply_prepared(self.right_plan(lp, b, c, self.depth))
                    partial = fnp.sum(c, axis=0)
                    partial = partial[:rows, :cols]
                    if tile0 == 0:
                        fnp.copyto(destination, partial)
                    else:
                        fnp.add(destination, partial, out=destination)
_adjusted_pooled_cp_engine = None

def _adjusted_pooled_cp_product(left, right):
    return _adjusted_pooled_cp_engine.product(left, right)

class AdjustedPooledCPSetup(AdjustedPooledCPPreparedLong):

    def reset_prediction(self):
        self.prepared_source = None
        self.prepared_key = None

    def warm_common_plans(self):
        cap = self.capacity
        if cap < self.min_size:
            return
        for inner in range(32, cap + 1, 32):
            a = self.buffer('ioA', (1, cap, inner), fnp.float32)
            lp = self.left_plan(a, self.depth)
            for count in range(1, self.batch + 1):
                b = self.buffer('ioB', (count, inner, cap), fnp.float32)
                c = self.buffer('ioC', (count, cap, cap), fnp.float32)
                self.right_plan(lp, b, c, self.depth)
                aa = self.buffer('ioA', (count, cap, inner), fnp.float32)
                self.right_plan(self.left_plan(aa, self.depth), b, c, self.depth)

def _adjusted_pooled_cp_engine_for_width(width):
    global _adjusted_pooled_cp_engine
    required = min(1024, max(32, (width + 31) // 32 * 32))
    if _adjusted_pooled_cp_engine is None or _adjusted_pooled_cp_engine.capacity < required:
        _adjusted_pooled_cp_engine = AdjustedRaggedRightPlanCP(depth=5, capacity=required, batch=8, min_size=128)
    _adjusted_pooled_cp_engine.reset_prediction()
    return _adjusted_pooled_cp_engine

class AdjustedPooledCPSetupCanonical(AdjustedPooledCPSetup):

    def warm_common_plans(self):
        cap = self.capacity
        if cap < self.min_size:
            return
        dtype = fnp.empty((0,), dtype=fnp.float32).dtype
        for inner in range(32, cap + 1, 32):
            a = self.buffer('ioA', (1, cap, inner), dtype)
            lp = self.left_plan(a, self.depth)
            for count in range(1, self.batch + 1):
                b = self.buffer('ioB', (count, inner, cap), dtype)
                c = self.buffer('ioC', (count, cap, cap), dtype)
                self.right_plan(lp, b, c, self.depth)
                aa = self.buffer('ioA', (count, cap, inner), dtype)
                self.right_plan(self.left_plan(aa, self.depth), b, c, self.depth)


class AdjustedCopyBatchedPooledCP(AdjustedPooledCPSetupCanonical):
    """Batch canonical identity-slot copies without changing floating arithmetic.

    Public stack(out=...) writes directly into adjacent basic-slice slot views
    of the canonical pool. Every nontrivial Winograd addition, subtraction,
    leaf product, reconstruction, padding write and tile partial sum retains
    its original operands and evaluation order. Scratch capacity, setup and
    prediction reset are inherited unchanged; no new numeric cache is added.
    """

    def prepare_left(self, plan):
        a, stage, child = plan
        if stage is None:
            return
        (a11, a12, a21, a22), av = stage
        fnp.stack((a11, a12), axis=0, out=child[0][:2])
        fnp.add(a21, a22, out=av[4])
        fnp.subtract(av[4], a11, out=av[5])
        fnp.subtract(a11, a21, out=av[6])
        fnp.subtract(a12, av[5], out=av[2])
        fnp.copyto(av[3], a22)
        self.prepare_left(child)

    def multiply_prepared(self, plan):
        a, b, out, stage = plan
        if stage is None:
            fnp.matmul(a, b, out=out)
            return
        (b11, b12, b21, b22), bv, pv, (c11, c12, c21, c22), child = stage
        fnp.stack((b11, b21, b22), axis=0, out=child[1][:3])
        fnp.subtract(b12, b11, out=bv[4])
        fnp.subtract(b22, bv[4], out=bv[5])
        fnp.subtract(b22, b12, out=bv[6])
        fnp.subtract(bv[5], b21, out=bv[3])
        self.multiply_prepared(child)
        fnp.add(pv[0], pv[1], out=c11)
        fnp.add(pv[0], pv[5], out=pv[0])
        fnp.add(pv[0], pv[6], out=pv[6])
        fnp.add(pv[0], pv[4], out=pv[0])
        fnp.add(pv[0], pv[2], out=c12)
        fnp.subtract(pv[6], pv[3], out=c21)
        fnp.add(pv[6], pv[4], out=c22)

_review_vector_cache = {}

def _review_derivative_stack(p, upto, sigma, phi, prob, hs, rs):
    key = ('derivative_stack', p, upto)
    if key not in _review_vector_cache:
        _review_vector_cache[key] = fnp.stack([derivative(p, j, sigma, phi, prob, hs, rs) for j in range(1, upto + 1)])
    return _review_vector_cache[key]

def _review_centered_square(k, sigma, phi, prob, hs, rs):
    upto = max(4, k)
    key = ('square', upto)
    if key not in _review_vector_cache:
        d1 = _review_derivative_stack(1, upto, sigma, phi, prob, hs, rs)
        d2 = _review_derivative_stack(2, upto, sigma, phi, prob, hs, rs)
        _review_vector_cache[key] = d2 - 2 * rs[1][None, :] * d1
    return _review_vector_cache[key][k - 1]

def _review_sigma_power(sigma, k):
    key = ('sigma_power', k)
    if key not in _review_vector_cache:
        _review_vector_cache[key] = sigma ** k
    return _review_vector_cache[key]

def _review_gaussian_k3(rs):
    key = 'gaussian_k3'
    if key not in _review_vector_cache:
        mean = rs[1]
        _review_vector_cache[key] = rs[3] - 3 * mean * rs[2] + 2 * mean ** 3
    return _review_vector_cache[key]

def _review_prime_native_derivative_cache(sigma, phi, prob, hs, rs):
    """Batch the fifteen default nonfinal derivative formulas; same per-element order."""
    high_keys = [(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (3, 4)]
    if any(key in _derivative_cache for key in high_keys):
        raise ValueError('Derivative priming requires the newly cleared layer cache.')
    factors = fnp.asarray([math.factorial(p) * (-1) ** (k - p - 1) for p, k in high_keys], dtype=sigma.dtype)
    h_table = fnp.stack([hs[k - p - 1] for p, k in high_keys])
    sigma_powers = fnp.stack([_review_sigma_power(sigma, k - p) for p, k in high_keys])
    high = factors[:, None] * h_table * phi[None, :] / sigma_powers
    low_keys = [(1, 1), (2, 1), (2, 2), (3, 1), (3, 2), (3, 3), (4, 3), (4, 4)]
    factors = fnp.asarray([math.factorial(p) / math.factorial(p - k) if k < p else math.factorial(p) for p, k in low_keys], dtype=sigma.dtype)
    low_inputs = fnp.stack([rs[p - k] if k < p else prob for p, k in low_keys])
    low = factors[:, None] * low_inputs
    for index, key in enumerate(high_keys):
        _derivative_cache[key] = high[index]
    for index, key in enumerate(low_keys):
        _derivative_cache[key] = low[index]

def _review_corrected_moment_block(sigma, phi, prob, hs, rs, skew, kurt):
    """Four scalar Edgeworth moments with the original left-to-right additions."""
    base = fnp.stack([rs[p] for p in range(1, 5)])
    third = fnp.stack([derivative(p, 3, sigma, phi, prob, hs, rs) for p in range(1, 5)])
    fourth = fnp.stack([derivative(p, 4, sigma, phi, prob, hs, rs) for p in range(1, 5)])
    values = base + skew[None, :] * third / 6 + kurt[None, :] * fourth / 24
    return [rs[0]] + [values[index] for index in range(4)]

def gaussian_third_factors(cp, sigma, phi, prob, hs, rs, degree=4):
    """Batch pair powers; remove discarded path/adjust matrices from degree-four branch."""
    if degree != 4:
        return _review_gaussian_third_factors_original(cp, sigma, phi, prob, hs, rs, degree)
    # Preserve this cache population: later native scalar feedback reuses it.
    _review_gaussian_k3(rs)
    rho = cp / (sigma[:, None] * sigma[None, :])
    fnp.fill_diagonal(rho, 0.0)
    power = fnp.ones_like(cp)
    powers, f_values, g_values, scales = ([], [], [], [])
    for k in range(1, 5):
        power = power * rho
        powers.append(power)
        f_values.append(derivative(1, k, sigma, phi, prob, hs, rs))
        g_values.append(_review_centered_square(k, sigma, phi, prob, hs, rs))
        scales.append(_review_sigma_power(sigma, k))
    power_table = fnp.stack(powers)
    f_table = fnp.stack(f_values)
    g_table = fnp.stack(g_values)
    scale_table = fnp.stack(scales)
    factors = fnp.asarray([math.factorial(k) for k in range(1, 5)], dtype=cp.dtype)
    scaled_g = g_table * scale_table
    scaled_f = f_table * scale_table
    terms = scaled_g[:, :, None] * scaled_f[:, None, :] * power_table / factors[:, None, None]
    pair = fnp.zeros_like(cp)
    for k in range(4):
        pair = pair + terms[k]
    h = phi / sigma
    b = prob[:, None] * cp
    a = fnp.diag(3 * h)
    return (a, b, pair)


class AdjustedWideColumnBatchedCP(AdjustedCopyBatchedPooledCP):
    """Use eight full column tiles, canonical four-slot tails, and four inner tiles.

    A wide product writes each tile independently. Two original four-slot groups
    are combined only if all eight column tiles are full, so per-tile shapes,
    padding, operands and the counted element work remain unchanged. Ragged tails
    follow the original four-slot grouping. Long-inner groups and sum/add order
    stay at four. The inherited shape-only scratch pool now reserves eight slots;
    warm plans cover wide counts one through eight and inner counts one through
    four. No model data, numerical result or cross-prediction cache is added.
    """

    def __init__(self, depth=5, capacity=1024, batch=8, min_size=128, min_block=16):
        if batch != 8:
            raise ValueError('This engine uses eight wide slots and four inner slots.')
        super().__init__(depth, capacity, batch, min_size, min_block)

    def _wide(self, left, right, output):
        m, k = left.shape
        n = right.shape[1]
        cap = self.capacity
        kp = (k + 31) // 32 * 32
        column_tiles = (n + cap - 1) // cap
        for row0 in range(0, m, cap):
            rows = min(cap, m - row0)
            rp = (rows + 31) // 32 * 32
            a = self.buffer('ioA', (1, rp, kp), left.dtype)
            lp = self.left_plan(a, self.depth)
            source_key = (row0, rows, k, rp, kp)
            if self.prepared_source is not left or self.prepared_key != source_key:
                if rows != rp or k != kp:
                    fnp.copyto(a, 0)
                fnp.copyto(a[0, :rows, :k], left[row0:row0 + rows, :])
                self.prepare_left(lp)
                self.prepared_source = left
                self.prepared_key = source_key
            tile0 = 0
            while tile0 < column_tiles:
                # Only merge two canonical groups when all eight columns are full.
                # Four-slot tail groups retain the original tile shapes and padding.
                count = self.batch if tile0 + self.batch <= n // cap else min(4, column_tiles - tile0)
                columns = min(cap, n - tile0 * cap)
                cp = (columns + 31) // 32 * 32
                b = self.buffer('ioB', (count, kp, cp), right.dtype)
                c = self.buffer('ioC', (count, rp, cp), output.dtype)
                bv, cv = (self.tile_slots(b), self.tile_slots(c))
                last_cols = min(cap, n - (tile0 + count - 1) * cap)
                if k != kp or last_cols != cp:
                    fnp.copyto(b, 0)
                for slot in range(count):
                    col0 = (tile0 + slot) * cap
                    cols = min(cap, n - col0)
                    fnp.copyto(self.cropped(bv[slot], k, cols), right[:, col0:col0 + cols])
                self.multiply_prepared(self.right_plan(lp, b, c, self.depth))
                for slot in range(count):
                    col0 = (tile0 + slot) * cap
                    cols = min(cap, n - col0)
                    fnp.copyto(output[row0:row0 + rows, col0:col0 + cols], self.cropped(cv[slot], rows, cols))
                tile0 += count

    def _long_inner(self, left, right, output):
        self.prepared_source = None
        self.prepared_key = None
        m, k = left.shape
        n = right.shape[1]
        cap = self.capacity
        inner_tiles = (k + cap - 1) // cap
        inner_batch = min(self.batch, 4)
        for row0 in range(0, m, cap):
            rows = min(cap, m - row0)
            rp = (rows + 31) // 32 * 32
            for col0 in range(0, n, cap):
                cols = min(cap, n - col0)
                cp = (cols + 31) // 32 * 32
                destination = output[row0:row0 + rows, col0:col0 + cols]
                for tile0 in range(0, inner_tiles, inner_batch):
                    count = min(inner_batch, inner_tiles - tile0)
                    inner = min(cap, k - tile0 * cap)
                    kp = (inner + 31) // 32 * 32
                    a, b, c, av, bv, cv = self._views(count, rp, kp, cp, left.dtype)
                    last_span = min(cap, k - (tile0 + count - 1) * cap)
                    if rows != rp or last_span != kp:
                        fnp.copyto(a, 0)
                    if cols != cp or last_span != kp:
                        fnp.copyto(b, 0)
                    for slot in range(count):
                        k0 = (tile0 + slot) * cap
                        span = min(cap, k - k0)
                        fnp.copyto(self.cropped(av[slot], rows, span), left[row0:row0 + rows, k0:k0 + span])
                        fnp.copyto(self.cropped(bv[slot], span, cols), right[k0:k0 + span, col0:col0 + cols])
                    lp = self.left_plan(a, self.depth)
                    self.prepare_left(lp)
                    self.multiply_prepared(self.right_plan(lp, b, c, self.depth))
                    partial = fnp.sum(c, axis=0)
                    partial = partial[:rows, :cols]
                    if tile0 == 0:
                        fnp.copyto(destination, partial)
                    else:
                        fnp.add(destination, partial, out=destination)

    def warm_common_plans(self):
        cap = self.capacity
        if cap < self.min_size:
            return
        dtype = fnp.empty((0,), dtype=fnp.float32).dtype
        for inner in range(32, cap + 1, 32):
            a = self.buffer('ioA', (1, cap, inner), dtype)
            lp = self.left_plan(a, self.depth)
            for count in range(1, self.batch + 1):
                b = self.buffer('ioB', (count, inner, cap), dtype)
                c = self.buffer('ioC', (count, cap, cap), dtype)
                self.right_plan(lp, b, c, self.depth)
                if count <= 4:
                    aa = self.buffer('ioA', (count, cap, inner), dtype)
                    self.right_plan(self.left_plan(aa, self.depth), b, c, self.depth)


class AdjustedRaggedRightPlanCP(AdjustedWideColumnBatchedCP):
    """Prepare every reachable padded ragged right plan for widths<=1024.

    CP transport/query products retain full model-width output rows, rounded
    to pool capacity. Multi-tile wide groups and all long-inner output columns
    use full capacity. Only a final singleton wide tile has a smaller padded
    column extent. Setup therefore prepares count1 at every inner/column
    multiple of32, and IO slot views for existing full-column counts1..4/8.
    The inherited scratch pool and all prediction arithmetic stay unchanged.
    Only shape metadata and views into that pool are cached; no model values
    or predictions are read or computed. Exact unrounded per-call crop views
    remain lazy because their full Cartesian family is much larger.
    """

    def warm_common_plans(self):
        super().warm_common_plans()
        cap = self.capacity
        if cap < self.min_size:
            return
        dtype = fnp.empty((0,), dtype=fnp.float32).dtype
        for inner in range(32, cap + 1, 32):
            a = self.buffer('ioA', (1, cap, inner), dtype)
            lp = self.left_plan(a, self.depth)
            # These full-column plans already exist; their per-tile IO views
            # are also shape-only and can be prepared before prediction.
            for count in (1, 2, 3, 4, 8):
                b = self.buffer('ioB', (count, inner, cap), dtype)
                c = self.buffer('ioC', (count, cap, cap), dtype)
                self.tile_slots(b)
                self.tile_slots(c)
            for columns in range(32, cap, 32):
                b = self.buffer('ioB', (1, inner, columns), dtype)
                c = self.buffer('ioC', (1, cap, columns), dtype)
                self.tile_slots(b)
                self.tile_slots(c)
                self.right_plan(lp, b, c, self.depth)

def _review_centered_cube_family(sigma, phi, prob, hs, rs, upto=4):
    """Share the exact cubic prefix; retain the two original final term orders."""
    key = ('centered_cube_family', upto)
    if key not in _review_vector_cache:
        d1 = _review_derivative_stack(1, upto, sigma, phi, prob, hs, rs)
        d2 = _review_derivative_stack(2, upto, sigma, phi, prob, hs, rs)
        d3 = _review_derivative_stack(3, upto, sigma, phi, prob, hs, rs)
        mu = rs[1]
        three_mu = 3 * mu
        prefix = d3 - three_mu[None, :] * d2
        # These products deliberately differ: no reassociation or power rewrite.
        birth = prefix + (three_mu * mu)[None, :] * d1
        directional = prefix + (3 * (mu ** 2))[None, :] * d1
        _review_vector_cache[key] = (birth, directional)
    return _review_vector_cache[key]
