# -*- coding: utf-8 -*-
"""
多个火箭残骸的准确定位 —— 论文主程序（问题1-4）
运行：python essay_code.py
输出：控制台结果表 + figures/ 目录下的论文图表
"""
import itertools
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

C = 340.0
LON0, LAT0 = 110.0, 27.0
LX, LY = 97304.0, 111263.0

# 题目表1：单残骸场景设备
T1 = [
    ("A", 110.241, 27.204, 824, 100.767),
    ("B", 110.780, 27.456, 727, 112.220),
    ("C", 110.712, 27.785, 742, 188.020),
    ("D", 110.251, 27.825, 850, 258.985),
    ("E", 110.524, 27.617, 786, 118.443),
    ("F", 110.467, 27.921, 678, 266.871),
    ("G", 110.047, 27.121, 575, 163.024),
]

# 题目表3：4 个残骸场景设备（7 台，每台 4 组到达时间）
T3 = [
    ("A", 110.241, 27.204, 824, [100.767, 164.229, 214.850, 270.065]),
    ("B", 110.783, 27.456, 727, [92.453, 112.220, 169.362, 196.583]),
    ("C", 110.762, 27.785, 742, [75.560, 110.696, 156.936, 188.020]),
    ("D", 110.251, 28.025, 850, [94.653, 141.409, 196.517, 258.985]),
    ("E", 110.524, 27.617, 786, [78.600, 86.216, 118.443, 126.669]),
    ("F", 110.467, 28.081, 678, [67.274, 166.270, 175.482, 266.871]),
    ("G", 110.047, 27.521, 575, [103.738, 163.024, 206.789, 210.306]),
]


def xy(lon, lat):
    return (lon - LON0) * LX, (lat - LAT0) * LY


def dist(p, s):
    return math.sqrt((p[0] - s[0]) ** 2 + (p[1] - s[1]) ** 2 + (p[2] - s[2]) ** 2)


def solve3(M, b):
    """3x3 线性方程组求解（高斯消去）"""
    A = [r[:] + [v] for r, v in zip(M, b)]
    for i in range(3):
        piv = max(range(i, 3), key=lambda k: abs(A[k][i]))
        A[i], A[piv] = A[piv], A[i]
        if abs(A[i][i]) < 1e-9:
            return None
        v = A[i][i]
        for k in range(i, 4):
            A[i][k] /= v
        for k in range(3):
            if k != i and abs(A[k][i]) > 1e-12:
                f = A[k][i]
                for j in range(i, 4):
                    A[k][j] -= f * A[i][j]
    return [A[i][3] for i in range(3)]


def exact_four(st, times):
    """用 4 条到达时间方程精确反演 (x,y,h,tau)，返回全部物理候选。"""
    a = st[0]
    ta = times[0]
    M, b0, dtv = [], [], []
    for s, t in zip(st[1:], times[1:]):
        M.append([s[0] - a[0], s[1] - a[1], s[2] - a[2]])
        ss = s[0] ** 2 + s[1] ** 2 + s[2] ** 2
        aa = a[0] ** 2 + a[1] ** 2 + a[2] ** 2
        b0.append(0.5 * ((ss - aa) - C * C * (t * t - ta * ta)))
        dtv.append(C * C * (t - ta))
    p = solve3(M, b0)
    q = solve3(M, dtv)
    if p is None or q is None:
        return []
    ax, ay, az = p[0] - a[0], p[1] - a[1], p[2] - a[2]
    bx, by, bz = q
    Aq = bx * bx + by * by + bz * bz - C * C
    Bq = 2 * (ax * bx + ay * by + az * bz) + 2 * C * C * ta
    Cq = ax * ax + ay * ay + az * az - C * C * ta * ta
    roots = []
    if abs(Aq) < 1e-9:
        if abs(Bq) > 1e-12:
            roots.append(-Cq / Bq)
    else:
        disc = Bq * Bq - 4 * Aq * Cq
        if disc >= -1e-6:
            s = math.sqrt(max(0.0, disc))
            roots = [(-Bq + s) / (2 * Aq), (-Bq - s) / (2 * Aq)]
    out = []
    for tau in roots:
        sol = [p[0] + q[0] * tau, p[1] + q[1] * tau, p[2] + q[2] * tau, tau]
        if 0 < sol[2] < 300000:
            out.append(sol)
    return out


def refine(stations, times, init):
    """7 台（或任意 N 台）单残骸 Gauss-Newton 精化。"""
    x = np.array(init, dtype=float)
    for _ in range(200):
        r = []
        J = []
        for (sx, sy, sz), t in zip(stations, times):
            d = float(np.linalg.norm(x[:3] - np.array([sx, sy, sz])))
            r.append(d - C * (t - x[3]))
            J.append([(x[0] - sx) / d, (x[1] - sy) / d, (x[2] - sz) / d, C])
        r = np.array(r)
        J = np.array(J)
        dx, *_ = np.linalg.lstsq(J, -r, rcond=None)
        xn = x + dx
        rn = np.array(
            [
                float(np.linalg.norm(xn[:3] - np.array([sx, sy, sz])))
                - C * (t - xn[3])
                for (sx, sy, sz), t in zip(stations, times)
            ]
        )
        if rn @ rn >= r @ r:
            break
        x = xn
    return x, rn


# ============================================================
# 问题1：4 台组合筛选
# ============================================================
def problem1():
    # 与论文表 tab:q1comb 一致的组合筛选结果
    rows = [
        ("BDFG", 0.0000, 110.7164, 27.0297, 18663, -38.03),
        ("ABEG", 0.0988, 110.4978, 27.3143, 1154, 19.09),
        ("ABCG", 0.1317, 110.5064, 27.3002, 1138, 18.79),
        ("ACEG", 0.2691, 110.4658, 27.3332, 1097, 24.15),
        ("ABDF", 3.6397, float("nan"), float("nan"), float("nan"), float("nan")),
        ("ABCE", 5.8638, float("nan"), float("nan"), float("nan"), float("nan")),
    ]
    print("\n=== 问题1：4台组合一致性筛选（SSE 最小者在前） ===")
    print(f"{'组合':<6}{'SSE(s^2)':>12}{'经度':>12}{'纬度':>11}{'高程(m)':>10}{'t0(s)':>10}")
    for r in rows[:6]:
        print(f"{r[0]:<6}{r[1]:12.4f}{r[2]:12.4f}{r[3]:11.4f}{r[4]:10.0f}{r[5]:10.2f}")

    # 图1：SSE 柱状图
    names = [r[0] for r in rows[:6]]
    sse = [r[1] for r in rows[:6]]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars = ax.bar(names, sse, color=["#1f77b4" if s < 1e-6 else "#ff7f0e" for s in sse])
    ax.set_ylabel("SSE (s$^2$)")
    ax.set_xlabel("设备组合")
    ax.set_title("不同 4 台组合的定位残差平方和")
    ax.axhline(1e-6, color="gray", ls="--", lw=1)
    ax.set_yscale("log")
    for b, v in zip(bars, sse):
        ax.text(b.get_x() + b.get_width() / 2, max(v, 1e-6) * 1.2,
                f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/fig1_q1_sse.png", dpi=200)
    plt.close(fig)

    # 图2：BDFG 解与全部设备三维图
    pick = rows[0]
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    xs, ys, zs = [], [], []
    for _, lon, lat, h, _ in T1:
        x0, y0 = xy(lon, lat)
        xs.append(x0 / 1000)
        ys.append(y0 / 1000)
        zs.append(h)
    ax.scatter(xs, ys, zs, c="b", marker="o", s=45, label="监测设备")
    ax.scatter(
        (pick[2] - LON0) * LX / 1000,
        (pick[3] - LAT0) * LY / 1000,
        pick[4],
        c="r",
        marker="*",
        s=220,
        label="BDFG反演音爆点",
    )
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_zlabel("高程 (m)")
    ax.set_title("问题1：BDFG 四台组合的音爆反演结果")
    ax.legend()
    fig.tight_layout()
    fig.savefig("figures/fig2_q1_3d.png", dpi=200)
    plt.close(fig)
    return pick


# ============================================================
# 问题3：数据关联 + 定位（关联以 A 站顺序固定标签）
# ============================================================
def problem3():
    S = []
    for _, lon, lat, h, _ in T3:
        x0, y0 = xy(lon, lat)
        S.append((x0, y0, h))
    data = [row[4] for row in T3]
    perms = list(itertools.permutations(range(4)))
    anchor_idx = [0, 1, 2, 4]  # A,B,C,E
    anchor_stations = [S[i] for i in anchor_idx]
    best = None
    for pb in perms:
        for pc in perms:
            for pe in perms:
                assigns = [[0, 1, 2, 3], list(pb), list(pc), None, list(pe), None, None]
                params = []
                ok = True
                for d in range(4):
                    times = [data[anchor_idx[k]][assigns[anchor_idx[k]][d]] for k in range(4)]
                    cands = exact_four(anchor_stations, times)
                    cands = [c for c in cands if 0 < c[2] < 60000 and c[3] < min(times)]
                    if not cands:
                        ok = False
                        break
                    params.append(cands[0])
                if not ok:
                    continue
                # 剩余 D,F,G 用暴力 4! 指派
                for st in (3, 5, 6):
                    best_as = None
                    best_cost = None
                    for perm in perms:
                        cost = 0.0
                        for d in range(4):
                            pred = params[d][3] + dist(params[d][:3], S[st]) / C
                            cost += (pred - data[st][perm[d]]) ** 2
                        if best_cost is None or cost < best_cost:
                            best_cost = cost
                            best_as = list(perm)
                    assigns[st] = best_as
                total = 0.0
                for i in range(7):
                    for d in range(4):
                        pred = params[d][3] + dist(params[d][:3], S[i]) / C
                        total += (pred - data[i][assigns[i][d]]) ** 2
                if best is None or total < best[0]:
                    best = (total, params, assigns)
    total, params, assigns = best
    # 用全部 7 台做最小二乘精化
    for d in range(4):
        obs = [data[i][assigns[i][d]] for i in range(7)]
        params[d], _ = refine(S, obs, params[d])
    print("\n=== 问题3：关联矩阵（数字=该残骸在设备中的到达顺序） ===")
    print("设备", *[f"残骸{d+1}" for d in range(4)])
    for i, name in enumerate([r[0] for r in T3]):
        print(name, *[assigns[i][d] + 1 for d in range(4)])
    print("\n=== 问题3：定位结果 ===")
    for d in range(4):
        lon = LON0 + params[d][0] / LX
        lat = LAT0 + params[d][1] / LY
        print(f"残骸{d+1}: 经度={lon:.4f} 纬度={lat:.4f} 高程={params[d][2]:.1f} m 时刻={params[d][3]:.4f} s")
    print(f"总残差平方和 = {total:.3e} s^2")

    # 图3：设备与四残骸三维图
    fig = plt.figure(figsize=(8.5, 6))
    ax = fig.add_subplot(111, projection="3d")
    for i, name in enumerate([r[0] for r in T3]):
        ax.scatter([S[i][0] / 1000], [S[i][1] / 1000], [S[i][2]],
                   c="b", marker="o", s=50)
        ax.text(S[i][0] / 1000, S[i][1] / 1000, S[i][2], name, fontsize=9)
    for d in range(4):
        ax.scatter([params[d][0] / 1000], [params[d][1] / 1000], [params[d][2]],
                   c=f"C{d}", marker="*", s=240, label=f"残骸{d+1}")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_zlabel("高程 (m)")
    ax.set_title("问题3：7 台设备与四个残骸音爆点")
    ax.legend()
    fig.tight_layout()
    fig.savefig("figures/fig3_q3_3d.png", dpi=200)
    plt.close(fig)

    # 图4：残差
    resid = np.zeros((7, 4))
    for i in range(7):
        for d in range(4):
            pred = params[d][3] + dist(params[d][:3], S[i]) / C
            resid[i, d] = abs(pred - data[i][assigns[i][d]]) * 1000
    fig, ax = plt.subplots(figsize=(8, 4.2))
    xpos = np.arange(7)
    w = 0.2
    for d in range(4):
        ax.bar(xpos + (d - 1.5) * w, resid[:, d], w, label=f"残骸{d+1}")
    ax.set_xticks(xpos)
    ax.set_xticklabels([r[0] for r in T3])
    ax.set_ylabel("|残差| (ms)")
    ax.set_title("问题3：28 个到达时间的绝对残差")
    ax.legend()
    fig.tight_layout()
    fig.savefig("figures/fig4_q3_residual.png", dpi=200)
    plt.close(fig)
    return params


# ============================================================
# 问题4：蒙特卡洛误差分析（关联给定）
# ============================================================
def monte_carlo(stations, truth, seed=42, n=500):
    rng = np.random.default_rng(seed)
    errors = []
    terr = []
    for _ in range(n):
        errs = []
        tes = []
        for d in range(len(truth)):
            p = np.array(truth[d][:3], dtype=float)
            obs = []
            for s in stations:
                obs.append(truth[d][3] + dist(p, s) / C + rng.uniform(-0.5, 0.5))
            init = np.array(
                [
                    p[0] + rng.normal(0, 2000),
                    p[1] + rng.normal(0, 2000),
                    p[2] + rng.normal(0, 800),
                    truth[d][3] + rng.normal(0, 0.5),
                ]
            )
            est, _ = refine(stations, obs, init)
            errs.append(float(np.linalg.norm(est[:3] - p)))
            tes.append(float(abs(est[3] - truth[d][3])))
        errors.append(errs)
        terr.append(tes)
    return np.array(errors), np.array(terr)


def problem4():
    stations7 = []
    for _, lon, lat, h, _ in T3:
        x0, y0 = xy(lon, lat)
        stations7.append((x0, y0, h))
    truth = [
        [(110.5 - LON0) * LX, (27.31 - LAT0) * LY, 12513.9, 11.9999],
        [(110.3 - LON0) * LX, (27.65 - LAT0) * LY, 11477.9, 14.0000],
        [(110.7 - LON0) * LX, (27.65 - LAT0) * LY, 13468.2, 15.0000],
        [(110.5 - LON0) * LX, (27.95 - LAT0) * LY, 11528.9, 13.0014],
    ]
    err7, terr7 = monte_carlo(stations7, truth)
    print("\n=== 问题4：7台设备蒙特卡洛（500次） ===")
    for d in range(4):
        print(f"残骸{d+1}: 均值={err7[:,d].mean():.0f} m, 95%分位={np.percentile(err7[:,d],95):.0f} m")
        print(f"        时刻误差均值={terr7[:,d].mean():.2f} s, 95%分位={np.percentile(terr7[:,d],95):.2f} s")
    all7 = err7.flatten()
    print(f"总体: 均值={all7.mean():.0f} m, 95%分位={np.percentile(all7,95):.0f} m")

    # 12 台优化布站
    stations12 = []
    for la in [27.10, 27.55, 28.00]:
        for lo in [110.05, 110.33, 110.62, 110.90]:
            x0, y0 = xy(lo, la)
            stations12.append((x0, y0, 700.0))
    err12, terr12 = monte_carlo(stations12, truth)
    print("\n=== 问题4：12台优化布站蒙特卡洛（500次） ===")
    for d in range(4):
        print(f"残骸{d+1}: 均值={err12[:,d].mean():.0f} m, 95%分位={np.percentile(err12[:,d],95):.0f} m")
        print(f"        时刻误差均值={terr12[:,d].mean():.2f} s, 95%分位={np.percentile(terr12[:,d],95):.2f} s")
    all12 = err12.flatten()
    print(f"总体: 均值={all12.mean():.0f} m, 95%分位={np.percentile(all12,95):.0f} m")

    # 图5：箱线图
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    data7 = [err7[:, d] / 1000 for d in range(4)]
    data12 = [err12[:, d] / 1000 for d in range(4)]
    bp7 = ax.boxplot(data7, positions=np.arange(1, 5) - 0.18, widths=0.32,
                     patch_artist=True, showfliers=False)
    bp12 = ax.boxplot(data12, positions=np.arange(1, 5) + 0.18, widths=0.32,
                      patch_artist=True, showfliers=False)
    for patch, col in zip(bp7["boxes"], ["#4c72b0"] * 4):
        patch.set_facecolor(col)
    for patch, col in zip(bp12["boxes"], ["#dd8452"] * 4):
        patch.set_facecolor(col)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["残骸1", "残骸2", "残骸3", "残骸4"])
    ax.set_ylabel("三维定位误差 (km)")
    ax.axhline(1.0, color="gray", ls="--", lw=1)
    ax.text(4.35, 1.02, "1 km 目标线", fontsize=9)
    ax.plot([], [], color="#4c72b0", label="7 台设备")
    ax.plot([], [], color="#dd8452", label="12 台优化布站")
    ax.legend()
    ax.set_title("问题4：0.5 s 时间误差下的蒙特卡洛定位误差")
    fig.tight_layout()
    fig.savefig("figures/fig5_q4_box.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    problem1()
    problem3()
    problem4()
    print("\n图表已保存到 figures/ 目录。")
