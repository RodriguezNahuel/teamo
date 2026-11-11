#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
teamo.py  -  una rosa hecha de letras que crece y florece en la terminal,
            con una carta que se abre al hacerle click.

Como ejecutarlo:
    py -m pip install numpy
    py teamo.py

Controles:
    click en el sobre  (o Enter / espacio / C)  ->  abrir / cerrar la carta
    rueda del mouse o flechas                   ->  desplazar la carta si es larga
    Q  o  Esc                                    ->  salir

El mismo archivo lo usa la version web (index.html): el navegador corre este
Python con Pyodide, asi que lo que escribas abajo aparece en los dos lados.
"""

# ╔═════════════════════════════ TU CARTA ═════════════════════════════╗
PARA = ""          # a quien va dirigida, ej: "Para vos"
CARTA = """
"""                # el texto completo de la carta (podes usar varios parrafos)
DE = ""            # la firma, ej: "Con amor, Nahuel"
FIRMA_HUD = "una rosa para vos · 2026"     # texto chiquito abajo a la izquierda
# ╚════════════════════════════════════════════════════════════════════╝

import math
import os
import sys
import time
import shutil
import textwrap
from functools import lru_cache

import numpy as np

TAU = 2 * np.pi

# ─────────────────────────── linea de tiempo ───────────────────────────
T_GROW = (0.4, 3.6)        # el tallo crece del suelo hasta la cabeza
T_BUD = (3.4, 1.4)         # aparece el capullo
T_BLOOM = (5.0, 6.5)       # los petalos se abren
T_SWAY = 7.0               # empieza a mecerse
T_LETTER = 9.5             # aparece el sobre

GROUND = -0.85
HEAD_Y = 0.22
HEAD_R = 0.47
TH_MIN, TH_MAX = 1.0 * np.pi, 15 * np.pi
CAM_D = 5.0
PITCH = 0.20
HEAD_TILT = 0.20

RAMP = np.array(list(".:-=+tfjzrxcvunoae0QB8&%$@"))
DUST_RAMP = np.array(list(".'+*"))

KEY = np.array([-0.35, 0.85, 0.65]) / np.linalg.norm([-0.35, 0.85, 0.65])
VIEW = np.array([0.0, 0.0, 1.0])


def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def norm(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def surf_normals(P):
    """normales de una superficie parametrica muestreada en una grilla (a, b, 3)"""
    return norm(np.cross(np.gradient(P, axis=0), np.gradient(P, axis=1)))


def grow(t):
    return float(smooth((t - T_GROW[0]) / T_GROW[1]))


def bud(t):
    return float(smooth((t - T_BUD[0]) / T_BUD[1]))


def bloom(t):
    return float(smooth((t - T_BLOOM[0]) / T_BLOOM[1]))


def grow_inv(s):
    u = np.linspace(0, 1, 2001)
    return float(u[np.searchsorted(smooth(u), s)])


def sway(t):
    amp = float(smooth((t - T_SWAY) / 4.0))
    return 0.42 * math.sin(max(t - T_SWAY, 0.0) * 0.35) * amp + 0.05 * math.sin(t * 0.21)


# ─────────────────────────── geometria ───────────────────────────
def stem_center(s):
    s = np.asarray(s, float)
    x = 0.04 * np.sin(s * np.pi * 1.25) - 0.01 * s
    y = GROUND + s * (HEAD_Y - GROUND)
    z = 0.03 * np.sin(s * np.pi)
    return np.stack([x, y, z], -1)


HEAD_BASE = stem_center(1.0)


@lru_cache(maxsize=4)
def rose_grid(nt, nx):
    th = np.linspace(TH_MIN, TH_MAX, nt)
    x = np.linspace(0, 1, nx)
    TH, XX = np.meshgrid(th, x, indexing="ij")
    THN = (TH - TH_MIN) / (TH_MAX - TH_MIN)
    # cada petalo: el borde se ondula con este perfil
    XU = 1 - 0.5 * (1.25 * (1 - np.mod(3.6 * TH, TAU) / np.pi) ** 2 - 0.25) ** 2
    petal = np.floor(3.6 * TH / TAU).astype(int)
    pv = np.random.default_rng(3).random(64)[petal % 64]
    tex = 0.5 + 0.5 * np.sin(TH * 9.7 + XX * 31.0) * np.cos(TH * 4.3 - XX * 17.0)
    ao = (0.45 + 0.55 * XX ** 0.55) * (1 - 0.35 * THN)
    return TH, XX, THN, XU, pv, tex, ao


def rot_x(P, a):
    c, s = math.cos(a), math.sin(a)
    out = P.copy()
    out[..., 1] = P[..., 1] * c - P[..., 2] * s
    out[..., 2] = P[..., 2] * c + P[..., 1] * s
    return out


def head_points(o, s_in, t, q):
    nt, nx = int(560 * q), int(18 * q)
    TH, XX, THN, XU, pv, tex, ao = rose_grid(nt, nx)
    o = o + 0.012 * math.sin(t * 1.3) * o            # respira un poquito
    m = 0.30 + 0.70 * o
    phi = (np.pi / 2) * np.exp(-TH / (8 * np.pi)) * m
    yb = 1.95653 * XX ** 2 * (1.27689 * XX - 1) ** 2 * np.sin(phi)
    r = XU * (XX * np.sin(phi) + yb * np.cos(phi))
    h = XU * (XX * np.cos(phi) - yb * np.sin(phi))
    h = h * (1 - 0.25 * THN * (1 - o))               # cerrado: los de afuera envuelven
    r = r * (0.80 + 0.20 * o) * (1 - (1 - o) * 0.75 * XX ** 2.2)    # capullo: se cierra arriba
    P = np.stack([r * np.sin(TH), h * (0.85 + 0.15 * o), r * np.cos(TH)], -1) * (HEAD_R * s_in)
    N = surf_normals(P)
    P = rot_x(P, HEAD_TILT) + HEAD_BASE
    N = rot_x(N, HEAD_TILT)
    k = P.shape[0] * P.shape[1]
    lo = np.stack([70 + 30 * pv, 0 * pv, 14 + 10 * pv], -1).reshape(k, 3)
    hi = np.stack([235 + 20 * pv, 35 + 25 * pv, 60 + 20 * pv], -1).reshape(k, 3)
    return dict(P=P.reshape(k, 3), N=N.reshape(k, 3), lo=lo, hi=hi,
                tex=tex.reshape(k), ao=ao.reshape(k), kind=np.zeros(k, int))


def green(k, bright=1.0):
    lo = np.tile([8.0, 60.0, 26.0], (k, 1))
    hi = np.tile([70.0, 245.0, 110.0], (k, 1)) * bright
    return lo, hi


def stem_points(g, q):
    ns, na = int(150 * q), 8
    s = np.linspace(0, 1, ns)
    a = np.linspace(0, TAU, na, endpoint=False)
    S, A = np.meshgrid(s, a, indexing="ij")
    S, A = S.ravel(), A.ravel()
    keep = S <= g
    S, A = S[keep], A[keep]
    rad = 0.024 * (1 - 0.2 * S)
    ring = np.stack([np.cos(A), np.zeros_like(A), np.sin(A)], -1)
    P = stem_center(S) + ring * rad[:, None]
    lo, hi = green(len(S))
    tex = 0.5 + 0.5 * np.sin(S * 90.0 + A * 3.0)
    return dict(P=P, N=ring, lo=lo, hi=hi, tex=tex, ao=np.ones(len(S)), kind=np.ones(len(S), int))


LEAVES = [  # s en el tallo, azimut, elevacion, largo, ancho
    (0.58, 0.20, 0.32, 0.34, 0.105),
    (0.30, np.pi - 0.15, 0.22, 0.38, 0.115),
]


def leaf_points(k, sc, q):
    s, az, el, L, W = LEAVES[k]
    el = el + (1 - sc) * 1.1
    B = stem_center(s)
    d = np.array([math.cos(el) * math.cos(az), math.sin(el), math.cos(el) * math.sin(az)])
    side = norm(np.cross([0.0, 0.0, 1.0], d))
    face = np.cross(d, side)
    nu, nv = int(34 * q), int(12 * q)
    U, V = np.meshgrid(np.linspace(0, 1, nu), np.linspace(-1, 1, nv), indexing="ij")
    Lr = L * sc
    w = W * sc * np.sin(np.pi * U ** 0.75) ** 0.85
    pet = 0.05 * sc
    P = (B + d * (pet + U * Lr)[..., None] + side * (V * w)[..., None]
         + face * (-0.35 * w * V ** 2)[..., None]
         + np.array([0.0, -1.0, 0.0]) * (0.22 * Lr * U ** 2)[..., None])
    N = surf_normals(P)
    n = nu * nv
    P, N = P.reshape(n, 3), N.reshape(n, 3)
    # peciolo
    u = np.linspace(0, 1, int(10 * q))[:, None]
    Pp = B + d * (pet * u)
    Np = np.tile(face, (len(u), 1))
    P = np.vstack([P, Pp])
    N = np.vstack([N, Np])
    lo, hi = green(len(P), 1.05)
    tex = np.concatenate([(0.5 + 0.5 * np.sin(U * 40 + V * 7)).ravel(), np.full(len(u), 0.5)])
    vein = np.concatenate([np.abs(V).ravel() < 0.12, np.ones(len(u), bool)])
    ao = np.where(vein, 0.75, 1.0)
    return dict(P=P, N=N, lo=lo, hi=hi, tex=tex, ao=ao, kind=np.ones(len(P), int))


def sepal_points(o, s_in, q):
    nu, nv = int(14 * q), 5
    U, V = np.meshgrid(np.linspace(0, 1, nu), np.linspace(-1, 1, nv), indexing="ij")
    Ps, Ns = [], []
    for i in range(5):
        a = i * TAU / 5 + 0.4
        rad = np.array([math.cos(a), 0.0, math.sin(a)])
        tan = np.array([-math.sin(a), 0.0, math.cos(a)])
        reach = (0.025 + (0.06 + 0.10 * o) * U) * s_in
        up = ((0.14 * (1 - o) - 0.15 * o) * U ** 1.3) * s_in
        w = 0.032 * (1 - U) ** 0.7 * s_in
        P = (HEAD_BASE + rad * reach[..., None] + np.array([0.0, 1.0, 0.0]) * up[..., None]
             + tan * (V * w)[..., None])
        Ps.append(P.reshape(-1, 3))
        Ns.append(surf_normals(P).reshape(-1, 3))
    P, N = np.vstack(Ps), np.vstack(Ns)
    lo, hi = green(len(P), 0.95)
    return dict(P=P, N=N, lo=lo, hi=hi, tex=np.full(len(P), 0.4), ao=np.ones(len(P)),
                kind=np.ones(len(P), int))


def plant(t, q):
    g = grow(t)
    parts = []
    if g > 0:
        parts.append(stem_points(g, q))
    for k, (s, *_rest) in enumerate(LEAVES):
        t_k = T_GROW[0] + T_GROW[1] * grow_inv(s)
        sc = float(smooth((t - t_k) / 1.3))
        if sc > 0.02:
            parts.append(leaf_points(k, sc, q))
    b = bud(t)
    if b > 0.01:
        o = bloom(t)
        parts.append(sepal_points(o, b, q))
        parts.append(head_points(o, b, t, q))
    if not parts:
        return None
    return {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}


# ─────────────────────────── camara ───────────────────────────
def to_camera(P, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    x = c * P[:, 0] + s * P[:, 2]
    z = -s * P[:, 0] + c * P[:, 2]
    y = P[:, 1]
    cp, sp = math.cos(PITCH), math.sin(PITCH)
    return np.stack([x, y * cp - z * sp, z * cp + y * sp], -1)


class View:
    """mapea coordenadas de camara a celdas de la grilla de texto"""

    def __init__(self, cols, rows, aspect, top, bottom):
        self.C, self.R = cols, rows
        self.top = top
        self.area = max(rows - top - bottom, 4)
        y_lo, y_hi = GROUND - 0.02, HEAD_Y + 0.56
        self.yc = 0.5 * (y_lo + y_hi)
        s_row = self.area * 0.94 / (y_hi - y_lo)
        s_col = s_row * aspect
        lim = cols * 0.96 / 1.30
        if s_col > lim:
            s_col = lim
            s_row = s_col / aspect
        self.sr, self.sc = s_row, s_col

    def project(self, Pc):
        f = CAM_D / (CAM_D - Pc[:, 2])
        col = self.C / 2 + Pc[:, 0] * f * self.sc
        row = self.top + self.area / 2 - (Pc[:, 1] - self.yc) * f * self.sr
        return col, row


def shift(a, dr, dc, fill):
    """a desplazado: out[r, c] = a[r + dr, c + dc]"""
    out = np.full_like(a, fill)
    R, C = a.shape[:2]
    rs = slice(max(0, -dr), min(R, R - dr))
    cs = slice(max(0, -dc), min(C, C - dc))
    rs2 = slice(max(0, dr), min(R, R + dr))
    cs2 = slice(max(0, dc), min(C, C + dc))
    out[rs, cs] = a[rs2, cs2]
    return out


# ─────────────────────────── destellos ───────────────────────────
SPK = np.random.default_rng(13).random((1600, 5))


def dust_points(t, head_center):
    """chispas que salen de la punta del tallo y brillitos alrededor de la flor"""
    D = SPK[:400]
    birth = T_GROW[0] + (T_GROW[1] + 0.8) * D[:, 0]
    age = t - birth
    life = 0.7 + 1.1 * D[:, 1]
    alive = (age > 0) & (age < life)
    tip = stem_center(smooth((birth - T_GROW[0]) / T_GROW[1]))
    ang = TAU * D[:, 2]
    el = (D[:, 3] - 0.5) * 1.6
    vel = np.stack([np.cos(ang) * np.sqrt(1 - el ** 2 / 4), el, np.sin(ang)], -1)
    vel *= (0.12 + 0.25 * D[:, 4])[:, None]
    a = np.clip(age, 0, None)[:, None]
    pos1 = tip + vel * a + np.array([0.0, -0.06, 0.0]) * a ** 2
    b1 = np.where(alive, 1 - age / life, 0.0)

    G = SPK[400:640]
    period = 1.5 + 2.5 * G[:, 0]
    ag = np.mod(t + G[:, 1] * period, period)
    b2 = 0.8 * np.sin(np.pi * ag / period) ** 4 * float(smooth((t - 5.5) / 3.0))
    ang = TAU * G[:, 2]
    rr = 0.45 + 0.55 * G[:, 3] ** 0.7
    off = np.stack([np.cos(ang) * rr, (G[:, 4] - 0.5) * 0.9, np.sin(ang) * rr * 0.6], -1)
    pos2 = head_center + off + np.array([0.0, 0.05, 0.0]) * ag[:, None]
    return np.vstack([pos1, pos2]), np.concatenate([b1, b2])


# ─────────────────────────── render ───────────────────────────
def render(t, cols, rows, aspect=2.0, top=1, bottom=4):
    """devuelve (chars[R,C], color[R,C,3] uint8, stats)"""
    C, R = cols, rows
    view = View(C, R, aspect, top, bottom)
    q = float(np.clip(view.sc / 55.0, 0.8, 2.4))
    yaw = sway(t)
    chars = np.full((R, C), " ", dtype="<U1")
    rgb = np.zeros((R, C, 3), float)
    occ = np.zeros((R, C), bool)

    data = plant(t, q)
    letters = 0
    if data is not None:
        Pc = to_camera(data["P"], yaw)
        Nc = to_camera(data["N"], yaw)
        Nc *= np.where(Nc[:, 2] < 0, -1.0, 1.0)[:, None]          # doble faz
        rim_l = norm(np.array([math.cos(t * 0.8), 0.4, math.sin(t * 0.8)]))
        half = norm(KEY + VIEW)
        diff = np.clip(Nc @ KEY, 0, 1)
        rim = np.clip(Nc @ rim_l, 0, 1)
        spec = np.clip(Nc @ half, 0, 1) ** 24
        fres = (1 - np.abs(Nc[:, 2])) ** 2
        lum = (0.18 + 0.72 * diff + 0.28 * rim + 0.22 * fres) * data["ao"]
        lum = np.clip(lum, 0, 1)
        color = data["lo"] + (data["hi"] - data["lo"]) * lum[:, None] + spec[:, None] * np.array([255.0, 190, 200]) * 0.55

        col, row = view.project(Pc)
        ci = np.floor(col).astype(int)
        ri = np.floor(row).astype(int)
        ok = (ci >= 0) & (ci < C) & (ri >= 0) & (ri < R)
        idx = (ri * C + ci)[ok]
        depth = Pc[ok, 2]
        order = np.lexsort((-depth, idx))
        idx_s = idx[order]
        first = np.ones(len(idx_s), bool)
        first[1:] = idx_s[1:] != idx_s[:-1]
        win = np.nonzero(ok)[0][order][first]
        cells = idx_s[first]

        D = np.full(R * C, -np.inf)
        D[cells] = Pc[win, 2]
        L = np.zeros(R * C)
        L[cells] = lum[win] * 0.85 + 0.15 * data["tex"][win]
        COL = np.zeros((R * C, 3))
        COL[cells] = color[win]
        D, L, COL = D.reshape(R, C), L.reshape(R, C), COL.reshape(R, C, 3)

        # tapar agujeritos de una celda
        for dr, dc in ((0, 1), (1, 0)):
            a_d, b_d = shift(D, -dr, -dc, -np.inf), shift(D, dr, dc, -np.inf)
            with np.errstate(invalid="ignore"):
                hole = ~np.isfinite(D) & np.isfinite(a_d) & np.isfinite(b_d) & (np.abs(a_d - b_d) < 0.05)
            if hole.any():
                D[hole] = a_d[hole]
                L[hole] = shift(L, -dr, -dc, 0)[hole]
                COL[hole] = shift(COL, -dr, -dc, 0)[hole]

        occ = np.isfinite(D)
        far = D[occ].min() - 1.0 if occ.any() else 0
        F = np.where(occ, D, far)
        thr, K = 0.06, 0.2

        def drop(dr, dc):
            return np.clip(F - shift(F, dr, dc, far), 0, K)

        dR, dL, dU, dD = drop(0, 1), drop(0, -1), drop(-1, 0), drop(1, 0)
        dUR, dUL, dDR, dDL = drop(-1, 1), drop(-1, -1), drop(1, 1), drop(1, -1)
        edge = occ & (np.maximum.reduce([dR, dL, dU, dD]) > thr)
        nx = dR - dL + 0.7 * (dUR + dDR - dUL - dDL)
        ny = dU - dD + 0.7 * (dUR + dUL - dDR - dDL)
        beta = np.degrees(np.arctan2(nx, -ny)) % 180.0           # angulo del borde
        ech = np.full((R, C), "-", dtype="<U1")
        ech[(beta >= 22.5) & (beta < 67.5)] = "/"
        ech[(beta >= 67.5) & (beta < 112.5)] = "|"
        ech[(beta >= 112.5) & (beta < 157.5)] = "\\"
        ech[((beta < 22.5) | (beta >= 157.5)) & (ny < 0)] = "_"

        ri_ = np.clip((L * (len(RAMP) - 1)).astype(int), 0, len(RAMP) - 1)
        chars[occ] = RAMP[ri_[occ]]
        chars[edge] = ech[edge]
        rgb[occ] = COL[occ]
        rgb[edge] = np.minimum(COL[edge] * 1.15 + 18, 255)
        letters = int(occ.sum())

    # ── cielo: estrellas, destellos grandes y polvo de diamante ──
    sky = np.zeros((R, C))
    S = SPK[1000:1180]
    vis = t > S[:, 2] * 4.0
    sx = (S[:, 0] * C).astype(int)
    sy = (S[:, 1] * R).astype(int)
    tw = 0.35 + 0.65 * (0.5 + 0.5 * np.sin(S[:, 3] * TAU + t * (1 + 3 * S[:, 4])))
    tw = tw * vis * (0.25 + 0.55 * S[:, 4] ** 2)
    np.maximum.at(sky, (np.clip(sy, 0, R - 1), np.clip(sx, 0, C - 1)), tw * 0.75)

    star = np.zeros((R, C))
    starch = np.full((R, C), " ", dtype="<U1")
    for i, (u, v, ph, sp, sz) in enumerate(SPK[1300:1311]):
        b = max(0.0, math.sin(TAU * (t / (2.5 + 3 * sp) + ph))) ** 5 * min(1.0, max(0.0, t - 1.0 - 4 * ph))
        if b < 0.05:
            continue
        cx, cy = int(u * C), int(top + v * (R - top - bottom))
        ln = int(1 + b * (3 + 4 * sz))
        for k in range(-ln, ln + 1):
            for (yy, xx, ch, fall) in ((cy, cx + k, "-", abs(k) / (ln + 1)),
                                       (cy + int(k / 2), cx, "|", abs(k) / (ln + 1))):
                if k == 0:
                    ch, fall = ("*" if b > 0.6 else "+"), 0.0
                if 0 <= yy < R and 0 <= xx < C:
                    val = b * (1 - fall) ** 1.5
                    if val > star[yy, xx]:
                        star[yy, xx] = val
                        starch[yy, xx] = ch

    head_c = HEAD_BASE + np.array([0.0, HEAD_R * 0.45, 0.0])
    pts, br = dust_points(t, head_c)
    m = br > 0.03
    if m.any():
        col, row = view.project(to_camera(pts[m], yaw))
        ci, ri = np.floor(col).astype(int), np.floor(row).astype(int)
        ok = (ci >= 0) & (ci < C) & (ri >= 0) & (ri < R)
        np.maximum.at(sky, (ri[ok], ci[ok]), br[m][ok])

    free = ~occ
    sm = free & (sky > 0.05) & (star < 0.05)
    di = np.clip((sky * len(DUST_RAMP)).astype(int), 0, len(DUST_RAMP) - 1)
    chars[sm] = DUST_RAMP[di[sm]]
    rgb[sm] = (sky[sm, None] * np.array([235.0, 225, 235]))
    st = free & (star >= 0.05)
    chars[st] = starch[st]
    rgb[st] = star[st, None] * np.array([255.0, 250, 255])

    stats = dict(bloom=bloom(t), sway=math.degrees(yaw * 9 + t * 0.9) % 360.0, letters=letters)
    return chars, np.clip(rgb, 0, 255).astype(np.uint8), stats


def web_frame(t, cols, rows, aspect, top, bottom):
    """usado por index.html (Pyodide)"""
    chars, rgb, st = render(t, cols, rows, aspect, top, bottom)
    return "".join(chars.ravel().tolist()), rgb.tobytes(), st["bloom"], st["sway"], st["letters"]


# ─────────────────────────── terminal ───────────────────────────
ENVELOPE = [
    " _________ ",
    "|\\       /|",
    "| \\  ♥  / |",
    "|  `---'  |",
    "|_________|",
]
ENV_LABEL = "tocá la carta"


class Term:
    def __init__(self):
        self.win = os.name == "nt"
        self.buf = ""
        self.ok_input = False
        if self.win:
            self._win_setup()
        else:
            self._unix_setup()
        out = "\x1b[?1049h\x1b[?25l\x1b[2J"
        if not self.win:
            out += "\x1b[?1000h\x1b[?1006h"
        sys.stdout.write(out)
        sys.stdout.flush()

    # windows: consola nativa (ReadConsoleInput) para mouse y teclado
    def _win_setup(self):
        import ctypes
        from ctypes import wintypes
        self.ct = ctypes
        k32 = ctypes.windll.kernel32
        self.k32 = k32
        self.hin = k32.GetStdHandle(-10)
        self.hout = k32.GetStdHandle(-11)
        mode = wintypes.DWORD()
        if k32.GetConsoleMode(self.hout, ctypes.byref(mode)):
            self.out_mode = mode.value
            k32.SetConsoleMode(self.hout, mode.value | 0x0004 | 0x0001)
        else:
            self.out_mode = None
        if k32.GetConsoleMode(self.hin, ctypes.byref(mode)):
            self.in_mode = mode.value
            new = (mode.value | 0x0010 | 0x0080 | 0x0008) & ~(0x0040 | 0x0002 | 0x0004 | 0x0200)
            k32.SetConsoleMode(self.hin, new)
            self.ok_input = True
        else:
            self.in_mode = None

        class COORD(ctypes.Structure):
            _fields_ = [("X", ctypes.c_short), ("Y", ctypes.c_short)]

        class KEY(ctypes.Structure):
            _fields_ = [("down", wintypes.BOOL), ("rep", wintypes.WORD), ("vk", wintypes.WORD),
                        ("scan", wintypes.WORD), ("ch", wintypes.WCHAR), ("ctrl", wintypes.DWORD)]

        class MOUSE(ctypes.Structure):
            _fields_ = [("pos", COORD), ("btn", wintypes.DWORD), ("ctrl", wintypes.DWORD),
                        ("flags", wintypes.DWORD)]

        class EV(ctypes.Union):
            _fields_ = [("key", KEY), ("mouse", MOUSE), ("pad", ctypes.c_byte * 16)]

        class REC(ctypes.Structure):
            _fields_ = [("type", wintypes.WORD), ("ev", EV)]

        self.REC = REC
        self.recs = (REC * 64)()

    def _unix_setup(self):
        try:
            import termios
            import tty
            self.fd = sys.stdin.fileno()
            self.old = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
            self.ok_input = True
        except Exception:
            self.old = None

    def events(self):
        """lista de ('click', x, y) / ('wheel', +-1) / ('key', ch)"""
        evs = []
        if not self.ok_input:
            return evs
        if self.win:
            n = self.ct.c_ulong()
            while self.k32.GetNumberOfConsoleInputEvents(self.hin, self.ct.byref(n)) and n.value:
                got = self.ct.c_ulong()
                self.k32.ReadConsoleInputW(self.hin, self.recs, 64, self.ct.byref(got))
                for i in range(got.value):
                    r = self.recs[i]
                    if r.type == 1 and r.ev.key.down:
                        vk, ch = r.ev.key.vk, r.ev.key.ch
                        if vk == 0x26:
                            evs.append(("wheel", 1))
                        elif vk == 0x28:
                            evs.append(("wheel", -1))
                        elif ch:
                            self.buf += ch
                    elif r.type == 2:
                        m = r.ev.mouse
                        if m.flags == 0 and (m.btn & 1):
                            evs.append(("click", m.pos.X, m.pos.Y))
                        elif m.flags == 4:
                            evs.append(("wheel", 1 if (m.btn >> 16) < 0x8000 else -1))
        else:
            import select
            while select.select([sys.stdin], [], [], 0)[0]:
                data = os.read(self.fd, 1024)
                if not data:
                    break
                self.buf += data.decode("utf-8", "ignore")
        evs += self._parse()
        return evs

    def _parse(self):
        evs, b = [], self.buf
        while b:
            if b.startswith("\x1b[<"):
                end = next((i for i, c in enumerate(b) if c in "Mm"), -1)
                if end < 0:
                    break
                try:
                    btn, x, y = (int(v) for v in b[3:end].split(";"))
                    if b[end] == "M":
                        if btn == 0:
                            evs.append(("click", x - 1, y - 1))
                        elif btn in (64, 65):
                            evs.append(("wheel", 1 if btn == 64 else -1))
                except ValueError:
                    pass
                b = b[end + 1:]
            elif b.startswith("\x1b[A") or b.startswith("\x1b[B"):
                evs.append(("wheel", 1 if b[2] == "A" else -1))
                b = b[3:]
            elif b.startswith("\x1b[") and len(b) >= 3:
                b = b[3:]
            elif b == "\x1b":
                evs.append(("key", "\x1b"))
                b = ""
            elif b[0] == "\x1b" and len(b) == 1:
                break
            else:
                evs.append(("key", b[0]))
                b = b[1:]
        self.buf = b
        return evs

    def write(self, s):
        sys.stdout.write(s)
        sys.stdout.flush()

    def close(self):
        out = "\x1b[0m\x1b[?25h\x1b[?1049l"
        if not self.win:
            out = "\x1b[?1000l\x1b[?1006l" + out
        sys.stdout.write(out)
        sys.stdout.flush()
        if self.win:
            if self.in_mode is not None:
                self.k32.SetConsoleMode(self.hin, self.in_mode)
            if self.out_mode is not None:
                self.k32.SetConsoleMode(self.hout, self.out_mode)
        elif getattr(self, "old", None) is not None:
            import termios
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)


def put(chars, fg, r, c, text, color, bg=None, bgarr=None):
    R, C = chars.shape
    if not (0 <= r < R):
        return
    for i, ch in enumerate(text):
        x = c + i
        if 0 <= x < C:
            chars[r, x] = ch
            fg[r, x] = color
            if bg is not None:
                bgarr[r, x] = bg


def letter_lines(width):
    lines = []
    if PARA.strip():
        lines += [PARA.strip(), ""]
    body = CARTA.strip("\n")
    if body.strip():
        for par in body.split("\n"):
            lines += textwrap.wrap(par, width) or [""]
    else:
        lines += [""] * 9
    if DE.strip():
        lines += ["", DE.strip().rjust(width)]
    return lines


def compose(t, chars, fg, st, env_box, letter, fps):
    """dibuja HUD, sobre y carta encima de la rosa. devuelve bg (o None por celda)"""
    R, C = chars.shape
    bg = np.zeros((R, C, 3), np.int16) - 1
    dim = np.array([120, 120, 128])
    if letter["open"]:
        fg[:] = (fg * 0.35).astype(np.uint8)

    put(chars, fg, 0, C - 14, "● t {:6.2f}s".format(t), dim)
    fg[0, C - 14] = (255, 70, 90)
    put(chars, fg, R - 4, 2, FIRMA_HUD, dim)
    put(chars, fg, R - 3, 2, "bloom {:5.1f}%   sway {:6.2f}°".format(st["bloom"] * 100, st["sway"]), dim)
    put(chars, fg, R - 2, 2, "letters {:,}".format(st["letters"]), dim)
    put(chars, fg, R - 1, 0, "$ py teamo.py", np.array([200, 200, 210]))
    right = "running · {:d} fps".format(int(round(fps)))
    put(chars, fg, R - 1, C - len(right) - 1, right, dim)

    # sobre
    env_box.clear()
    a = float(smooth((t - T_LETTER) / 1.2))
    if a > 0 and not letter["open"]:
        w = len(ENVELOPE[0])
        x0 = C - w - 3
        y0 = R - len(ENVELOPE) - 3
        pulse = 0.75 + 0.25 * math.sin(t * 3.0)
        paper = np.array([250, 225, 205]) * a * pulse
        for i, line in enumerate(ENVELOPE):
            put(chars, fg, y0 + i, x0, line, paper)
            j = line.find("♥")
            if j >= 0:
                fg[y0 + i, x0 + j] = np.array([255, 60, 85]) * a
        lab = ENV_LABEL
        put(chars, fg, y0 + len(ENVELOPE), x0 + (w - len(lab)) // 2, lab, np.array([255, 120, 140]) * a * pulse)
        env_box.update(x0=x0 - 1, x1=x0 + w + 1, y0=y0 - 1, y1=y0 + len(ENVELOPE) + 1)

    # carta abierta
    if letter["open"]:
        k = float(smooth((t - letter["at"]) / 0.45))
        W = min(C - 4, 66)
        inner = W - 6
        lines = letter_lines(inner)
        H_full = min(len(lines) + 6, R - 2)
        view_n = H_full - 6
        letter["max_scroll"] = max(0, len(lines) - view_n)
        letter["scroll"] = int(np.clip(letter["scroll"], 0, letter["max_scroll"]))
        H = max(2, int(round(H_full * k)))
        x0 = (C - W) // 2
        y0 = (R - H) // 2
        paper = np.array([250, 241, 228])
        ink = np.array([70, 30, 40])
        edge_c = np.array([180, 60, 80])
        for r in range(H):
            put(chars, fg, y0 + r, x0, " " * W, ink, paper, bg)
        put(chars, fg, y0, x0, "╭" + "─" * (W - 2) + "╮", edge_c, paper, bg)
        put(chars, fg, y0 + H - 1, x0, "╰" + "─" * (W - 2) + "╯", edge_c, paper, bg)
        for r in range(1, H - 1):
            put(chars, fg, y0 + r, x0, "│", edge_c, paper, bg)
            put(chars, fg, y0 + r, x0 + W - 1, "│", edge_c, paper, bg)
        if H >= 6:
            title = " ♥ "
            put(chars, fg, y0, x0 + (W - len(title)) // 2, title, np.array([220, 40, 70]), paper, bg)
            shown = lines[letter["scroll"]:letter["scroll"] + min(view_n, H - 6)]
            for i, ln in enumerate(shown):
                put(chars, fg, y0 + 3 + i, x0 + 3, ln, ink, paper, bg)
            if not "".join(lines).strip():
                rule = np.array([225, 205, 200])
                for i in range(min(view_n, H - 6)):
                    put(chars, fg, y0 + 3 + i, x0 + 3, "·" * inner, rule, paper, bg)
            hint = " click para cerrar "
            if letter["max_scroll"]:
                hint = " ↑↓ / rueda · click para cerrar "
            put(chars, fg, y0 + H - 1, x0 + (W - len(hint)) // 2, hint, np.array([160, 120, 125]), paper, bg)
    return bg


def to_ansi(chars, fg, bg):
    R, C = chars.shape
    out = ["\x1b[H"]
    fgl = fg.tolist()
    bgl = bg.tolist()
    chl = chars.tolist()
    for r in range(R):
        last_fg = last_bg = None
        row = []
        for c in range(C):
            ch = chl[r][c]
            b = bgl[r][c]
            if b[0] >= 0:
                if b != last_bg:
                    row.append("\x1b[48;2;%d;%d;%dm" % tuple(b))
                    last_bg = b
            elif last_bg is not None:
                row.append("\x1b[49m")
                last_bg = None
            if ch != " ":
                f = fgl[r][c]
                if f != last_fg:
                    row.append("\x1b[38;2;%d;%d;%dm" % tuple(f))
                    last_fg = f
            row.append(ch)
        row.append("\x1b[0m")
        out.append("".join(row))
        if r < R - 1:
            out.append("\n")
    return "".join(out)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    term = Term()
    letter = dict(open=False, at=0.0, scroll=0, max_scroll=0)
    env_box = {}
    t0 = time.perf_counter()
    last_size = None
    fps, frame_dt = 30.0, 1 / 30
    try:
        while True:
            tf = time.perf_counter()
            t = tf - t0
            for ev in term.events():
                if ev[0] == "key":
                    ch = ev[1].lower()
                    if ch in ("q", "\x1b", "\x03"):
                        return
                    if ch in ("\r", "\n", " ", "c") and t > T_LETTER:
                        letter["open"] = not letter["open"]
                        letter["at"], letter["scroll"] = t, 0
                elif ev[0] == "click":
                    _, x, y = ev
                    if letter["open"]:
                        letter["open"] = False
                    elif env_box and env_box["x0"] <= x <= env_box["x1"] and env_box["y0"] <= y <= env_box["y1"]:
                        letter["open"], letter["at"], letter["scroll"] = True, t, 0
                elif ev[0] == "wheel" and letter["open"]:
                    letter["scroll"] -= ev[1]

            size = shutil.get_terminal_size((100, 40))
            C, R = max(size.columns, 40), max(size.lines, 20)
            if size != last_size:
                term.write("\x1b[2J")
                last_size = size
            chars, fg, st = render(t, C, R, aspect=2.0, top=1, bottom=4)
            bg = compose(t, chars, fg, st, env_box, letter, fps)
            term.write(to_ansi(chars, fg, bg))

            spent = time.perf_counter() - tf
            if spent < frame_dt:
                time.sleep(frame_dt - spent)
            fps = 0.9 * fps + 0.1 / max(time.perf_counter() - tf, 1e-3)
    except KeyboardInterrupt:
        pass
    finally:
        term.close()


if __name__ == "__main__":
    main()
