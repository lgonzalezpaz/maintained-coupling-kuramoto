#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
omml_equations.py
=================
Constructs native Microsoft Word Office Math Markup Language (OMML, `<m:oMathPara>`)
elements for all mathematical equations (1)-(17) and (S1)-(S4) in the manuscript,
Supplementary Material, and Response to Reviewers.

Features physics/mathematics journal typography (Physical Review / JTB style):
- Cambria Math 12 pt display equations
- True stacked fractions (`<m:f>`)
- True N-ary summations (`∑`) and integrals (`∫`) with lower and upper limits (`<m:nary>`)
- Stretchy delimiters `( ... )`, `[ ... ]`, `| ... |`, `⟨ ... ⟩` (`<m:d>`)
- Upright roman operators and multi-letter subscripts (`d`, `i`, `exp`, `sin`, `cos`, `min`, `max`, `ln`, `Tr`, `ITE`, `RMST`, `eff`, `global`, `cross`, `base`, `space`, `pop`)
- Italic single-letter variables and Greek symbols
- Over-accents (`Ŝ`, `W̃`, `K̄`) via `<m:acc>`
"""

from __future__ import annotations
from lxml import etree

M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NSMAP = {"m": M_NS, "w": W_NS}


def _m(tag: str, **kwargs) -> etree._Element:
    el = etree.Element(f"{{{M_NS}}}{tag}", nsmap=NSMAP)
    for k, v in kwargs.items():
        el.set(f"{{{M_NS}}}{k}", v)
    return el


def _w(tag: str, **kwargs) -> etree._Element:
    el = etree.Element(f"{{{W_NS}}}{tag}", nsmap=NSMAP)
    for k, v in kwargs.items():
        el.set(f"{{{W_NS}}}{k}", v)
    return el


def m_run(text: str, upright: bool = False, bold: bool = False, size_half_pt: int = 24) -> etree._Element:
    """Create an OMML math run `<m:r>` with Cambria Math font."""
    r = _m("r")
    mrPr = _m("rPr")
    if upright and bold:
        mrPr.append(_m("sty", val="b"))
    elif upright:
        mrPr.append(_m("nor"))
        mrPr.append(_m("sty", val="p"))
    elif bold:
        mrPr.append(_m("sty", val="bi"))
    r.append(mrPr)

    wrPr = _w("rPr")
    rFonts = _w("rFonts")
    rFonts.set(f"{{{W_NS}}}ascii", "Cambria Math")
    rFonts.set(f"{{{W_NS}}}hAnsi", "Cambria Math")
    rFonts.set(f"{{{W_NS}}}cs", "Cambria Math")
    wrPr.append(rFonts)
    sz = _w("sz")
    sz.set(f"{{{W_NS}}}val", str(size_half_pt))
    szCs = _w("szCs")
    szCs.set(f"{{{W_NS}}}val", str(size_half_pt))
    wrPr.append(sz)
    wrPr.append(szCs)
    if bold:
        wrPr.append(_w("b"))
    r.append(wrPr)

    t = _m("t")
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    t.text = text
    r.append(t)
    return r


def _append_items(container: etree._Element, items):
    if isinstance(items, (list, tuple)):
        for it in items:
            container.append(it)
    else:
        container.append(items)


def m_sub(base, sub) -> etree._Element:
    sSub = _m("sSub")
    e = _m("e")
    _append_items(e, base)
    s = _m("sub")
    _append_items(s, sub)
    sSub.append(e)
    sSub.append(s)
    return sSub


def m_sup(base, sup) -> etree._Element:
    sSup = _m("sSup")
    e = _m("e")
    _append_items(e, base)
    s = _m("sup")
    _append_items(s, sup)
    sSup.append(e)
    sSup.append(s)
    return sSup


def m_subsup(base, sub, sup) -> etree._Element:
    sSubSup = _m("sSubSup")
    e = _m("e")
    _append_items(e, base)
    sb = _m("sub")
    _append_items(sb, sub)
    sp = _m("sup")
    _append_items(sp, sup)
    sSubSup.append(e)
    sSubSup.append(sb)
    sSubSup.append(sp)
    return sSubSup


def m_frac(num, den, bar_type: str = "bar") -> etree._Element:
    f = _m("f")
    fPr = _m("fPr")
    fPr.append(_m("type", val=bar_type))
    f.append(fPr)
    n = _m("num")
    _append_items(n, num)
    d = _m("den")
    _append_items(d, den)
    f.append(n)
    f.append(d)
    return f


def m_delim(inner, beg: str = "(", end: str = ")") -> etree._Element:
    d = _m("d")
    dPr = _m("dPr")
    dPr.append(_m("begChr", val=beg))
    dPr.append(_m("endChr", val=end))
    d.append(dPr)
    e = _m("e")
    _append_items(e, inner)
    d.append(e)
    return d


def m_nary(symbol: str, sub, sup, body, lim_loc: str = "undOvr") -> etree._Element:
    nary = _m("nary")
    naryPr = _m("naryPr")
    naryPr.append(_m("chr", val=symbol))
    naryPr.append(_m("limLoc", val=lim_loc))
    if sub is None:
        naryPr.append(_m("subHide", val="1"))
    if sup is None:
        naryPr.append(_m("supHide", val="1"))
    nary.append(naryPr)
    sb = _m("sub")
    if sub is not None:
        _append_items(sb, sub)
    sp = _m("sup")
    if sup is not None:
        _append_items(sp, sup)
    e = _m("e")
    _append_items(e, body)
    nary.append(sb)
    nary.append(sp)
    nary.append(e)
    return nary


def m_acc(base, chr_val: str = "\u0302") -> etree._Element:
    acc = _m("acc")
    accPr = _m("accPr")
    accPr.append(_m("chr", val=chr_val))
    acc.append(accPr)
    e = _m("e")
    _append_items(e, base)
    acc.append(e)
    return acc


def wrap_omath_para(elements: list[etree._Element]) -> etree._Element:
    oMathPara = _m("oMathPara")
    oMathParaPr = _m("oMathParaPr")
    oMathParaPr.append(_m("jc", val="center"))
    oMathPara.append(oMathParaPr)
    oMath = _m("oMath")
    for el in elements:
        oMath.append(el)
    oMathPara.append(oMath)
    return oMathPara


# ---------------------------------------------------------------------------
# Exact OMML Builders for Equations (1) - (17) and (S1) - (S4)
# ---------------------------------------------------------------------------

def build_equation_omml(eq_num: str) -> etree._Element | None:
    """Return an `<m:oMathPara>` element for the given equation number, e.g., '(1)' or '(S1)'."""
    key = eq_num.strip()

    if key == "(1)":
        # d_{ij} = \min(|x_i - x_j|, 1 - |x_i - x_j|)
        abs_x = m_delim([m_sub(m_run("x"), m_run("i")), m_run(" − "), m_sub(m_run("x"), m_run("j"))], "|", "|")
        return wrap_omath_para([
            m_sub(m_run("d"), m_run("ij")),
            m_run(" = "),
            m_run("min", upright=True),
            m_delim([abs_x, m_run(",  1 − "), m_delim([m_sub(m_run("x"), m_run("i")), m_run(" − "), m_sub(m_run("x"), m_run("j"))], "|", "|")], "(", ")"),
        ])

    if key == "(2)":
        # P_{ij}(t) = \frac{\rho(t)}{Z} \exp\left(-\frac{d_{ij}}{\lambda_{\mathrm{space}}}\right)
        return wrap_omath_para([
            m_sub(m_run("P"), m_run("ij")),
            m_delim(m_run("t"), "(", ")"),
            m_run(" = "),
            m_frac([m_run("ρ"), m_delim(m_run("t"), "(", ")")], m_run("Z")),
            m_run(" "),
            m_run("exp", upright=True),
            m_delim([
                m_run("−"),
                m_frac(m_sub(m_run("d"), m_run("ij")), m_sub(m_run("λ"), m_run("space", upright=True))),
            ], "(", ")"),
        ])

    if key == "(3)":
        # A_{\mathrm{base},ij} = K_0 \exp\left(-\frac{d_{ij}}{\lambda_{\mathrm{space}}}\right) \eta_{ij}
        return wrap_omath_para([
            m_sub(m_run("A"), [m_run("base", upright=True), m_run(",ij")]),
            m_run(" = "),
            m_sub(m_run("K"), m_run("0", upright=True)),
            m_run(" "),
            m_run("exp", upright=True),
            m_delim([
                m_run("−"),
                m_frac(m_sub(m_run("d"), m_run("ij")), m_sub(m_run("λ"), m_run("space", upright=True))),
            ], "(", ")"),
            m_run(" "),
            m_sub(m_run("η"), m_run("ij")),
        ])

    if key == "(4)":
        # \mathrm{d}\theta_i = \left[ \omega_i + \frac{1}{N_{\mathrm{pop}}} \sum_{j=1}^{N} A_{ij}(t) \sin(\theta_j - \theta_i) + I_{\mathrm{ext},i}(t) \right] \mathrm{d}t + \sigma \mathrm{d}W_i(t)
        summand = [
            m_sub(m_run("A"), m_run("ij")),
            m_delim(m_run("t"), "(", ")"),
            m_run(" "),
            m_run("sin", upright=True),
            m_delim([m_sub(m_run("θ"), m_run("j")), m_run(" − "), m_sub(m_run("θ"), m_run("i"))], "(", ")"),
        ]
        return wrap_omath_para([
            m_run("d", upright=True),
            m_sub(m_run("θ"), m_run("i")),
            m_run(" = "),
            m_delim([
                m_sub(m_run("ω"), m_run("i")),
                m_run(" + "),
                m_frac(m_run("1", upright=True), m_sub(m_run("N"), m_run("pop", upright=True))),
                m_run(" "),
                m_nary("∑", [m_run("j"), m_run("=", upright=True), m_run("1", upright=True)], m_run("N"), summand),
                m_run(" + "),
                m_sub(m_run("I"), [m_run("ext", upright=True), m_run(",i")]),
                m_delim(m_run("t"), "(", ")"),
            ], "[", "]"),
            m_run(" "),
            m_run("d", upright=True),
            m_run("t + σ "),
            m_run("d", upright=True),
            m_sub(m_run("W"), m_run("i")),
            m_delim(m_run("t"), "(", ")"),
        ])

    if key == "(5)":
        # A_{ij}(t) = A_{\mathrm{base},ij} + K_{\mathrm{cross},ij}(t) C_{ij}
        return wrap_omath_para([
            m_sub(m_run("A"), m_run("ij")),
            m_delim(m_run("t"), "(", ")"),
            m_run(" = "),
            m_sub(m_run("A"), [m_run("base", upright=True), m_run(",ij")]),
            m_run(" + "),
            m_sub(m_run("K"), [m_run("cross", upright=True), m_run(",ij")]),
            m_delim(m_run("t"), "(", ")"),
            m_run(" "),
            m_sub(m_run("C"), m_run("ij")),
        ])

    if key == "(6)":
        # \frac{\mathrm{d}K_{\mathrm{cross},ij}}{\mathrm{d}t} = \eta \cos(\theta_j - \theta_i) - \lambda_K K_{\mathrm{cross},ij}
        return wrap_omath_para([
            m_frac(
                [m_run("d", upright=True), m_sub(m_run("K"), [m_run("cross", upright=True), m_run(",ij")])],
                [m_run("d", upright=True), m_run("t")],
            ),
            m_run(" = η "),
            m_run("cos", upright=True),
            m_delim([m_sub(m_run("θ"), m_run("j")), m_run(" − "), m_sub(m_run("θ"), m_run("i"))], "(", ")"),
            m_run(" − "),
            m_sub(m_run("λ"), m_run("K")),
            m_run(" "),
            m_sub(m_run("K"), [m_run("cross", upright=True), m_run(",ij")]),
        ])

    if key == "(7)":
        # O_{\mu\nu} = \frac{1}{N} \left| \sum_{j=1}^{N} \exp\left[ \mathrm{i} (\theta_j^{(\mu)} - \theta_j^{(\nu)}) \right] \right|
        inner_exp = [
            m_run("exp", upright=True),
            m_delim([
                m_run("i", upright=True),
                m_delim([
                    m_subsup(m_run("θ"), m_run("j"), m_run("(μ)")),
                    m_run(" − "),
                    m_subsup(m_run("θ"), m_run("j"), m_run("(ν)")),
                ], "(", ")"),
            ], "[", "]"),
        ]
        return wrap_omath_para([
            m_sub(m_run("O"), m_run("μν")),
            m_run(" = "),
            m_frac(m_run("1", upright=True), m_run("N")),
            m_run(" "),
            m_delim([
                m_nary("∑", [m_run("j"), m_run("=", upright=True), m_run("1", upright=True)], m_run("N"), inner_exp),
            ], "|", "|"),
        ])

    if key == "(8)":
        # \langle \cos(\theta_j - \theta_i) \rangle_{\tau} = \frac{1}{\tau} \int_{t-\tau}^{t} \cos[\theta_j(s) - \theta_i(s)] \mathrm{d}s
        avg_lhs = m_sub(
            m_delim([
                m_run("cos", upright=True),
                m_delim([m_sub(m_run("θ"), m_run("j")), m_run(" − "), m_sub(m_run("θ"), m_run("i"))], "(", ")"),
            ], "⟨", "⟩"),
            m_run("τ"),
        )
        integrand = [
            m_run("cos", upright=True),
            m_delim([
                m_sub(m_run("θ"), m_run("j")),
                m_delim(m_run("s"), "(", ")"),
                m_run(" − "),
                m_sub(m_run("θ"), m_run("i")),
                m_delim(m_run("s"), "(", ")"),
            ], "[", "]"),
            m_run(" "),
            m_run("d", upright=True),
            m_run("s"),
        ]
        return wrap_omath_para([
            avg_lhs,
            m_run(" = "),
            m_frac(m_run("1", upright=True), m_run("τ")),
            m_run(" "),
            m_nary("∫", m_run("t−τ"), m_run("t"), integrand, lim_loc="subSup"),
        ])

    if key == "(9)":
        # P_{ij}^{\mathrm{create}} = \alpha_{\mathrm{create}} H( \langle \cos(\theta_j - \theta_i) \rangle_{\tau} - \theta_{\mathrm{thr}}^{\mathrm{create}} )
        avg_term = m_sub(
            m_delim([
                m_run("cos", upright=True),
                m_delim([m_sub(m_run("θ"), m_run("j")), m_run(" − "), m_sub(m_run("θ"), m_run("i"))], "(", ")"),
            ], "⟨", "⟩"),
            m_run("τ"),
        )
        return wrap_omath_para([
            m_subsup(m_run("P"), m_run("ij"), m_run("create", upright=True)),
            m_run(" = "),
            m_sub(m_run("α"), m_run("create", upright=True)),
            m_run(" "),
            m_run("H"),
            m_delim([
                avg_term,
                m_run(" − "),
                m_subsup(m_run("θ"), m_run("thr", upright=True), m_run("create", upright=True)),
            ], "(", ")"),
        ])

    if key == "(10)":
        # P_{ij}^{\mathrm{delete}} = \alpha_{\mathrm{delete}} H( \theta_{\mathrm{thr}}^{\mathrm{delete}} - \langle \cos(\theta_j - \theta_i) \rangle_{\tau} )
        avg_term = m_sub(
            m_delim([
                m_run("cos", upright=True),
                m_delim([m_sub(m_run("θ"), m_run("j")), m_run(" − "), m_sub(m_run("θ"), m_run("i"))], "(", ")"),
            ], "⟨", "⟩"),
            m_run("τ"),
        )
        return wrap_omath_para([
            m_subsup(m_run("P"), m_run("ij"), m_run("delete", upright=True)),
            m_run(" = "),
            m_sub(m_run("α"), m_run("delete", upright=True)),
            m_run(" "),
            m_run("H"),
            m_delim([
                m_subsup(m_run("θ"), m_run("thr", upright=True), m_run("delete", upright=True)),
                m_run(" − "),
                avg_term,
            ], "(", ")"),
        ])

    if key == "(11)":
        # E(t_2) \subset E(t_1) \quad \forall t_2 > t_1
        return wrap_omath_para([
            m_run("E"),
            m_delim(m_sub(m_run("t"), m_run("2", upright=True)), "(", ")"),
            m_run(" ⊂ E"),
            m_delim(m_sub(m_run("t"), m_run("1", upright=True)), "(", ")"),
            m_run("    ∀ "),
            m_sub(m_run("t"), m_run("2", upright=True)),
            m_run(" > "),
            m_sub(m_run("t"), m_run("1", upright=True)),
        ])

    if key == "(12)":
        # C_{\mathrm{binary}} = \frac{1}{N} \sum_{i=1}^{N} \frac{(B^3)_{ii}}{k_i(k_i-1)}, \quad C_{\mathrm{weighted}} = \frac{1}{N} \sum_{i=1}^{N} \frac{(\tilde{W}^{[1/3]})^3_{ii}}{k_i(k_i-1)}
        den_k = [m_sub(m_run("k"), m_run("i")), m_delim([m_sub(m_run("k"), m_run("i")), m_run(" − 1")], "(", ")")]
        num_bin = m_sub(m_delim(m_sup(m_run("B"), m_run("3", upright=True)), "(", ")"), m_run("ii"))
        w_tilde = m_acc(m_run("W"), "\u0303")
        num_wgt = m_subsup(
            m_delim(m_sup(w_tilde, m_run("[1/3]", upright=True)), "(", ")"),
            m_run("ii"),
            m_run("3", upright=True),
        )
        return wrap_omath_para([
            m_sub(m_run("C"), m_run("binary", upright=True)),
            m_run(" = "),
            m_frac(m_run("1", upright=True), m_run("N")),
            m_run(" "),
            m_nary("∑", [m_run("i"), m_run("=", upright=True), m_run("1", upright=True)], m_run("N"), m_frac(num_bin, den_k)),
            m_run(" ,      "),
            m_sub(m_run("C"), m_run("weighted", upright=True)),
            m_run(" = "),
            m_frac(m_run("1", upright=True), m_run("N")),
            m_run(" "),
            m_nary("∑", [m_run("i"), m_run("=", upright=True), m_run("1", upright=True)], m_run("N"), m_frac(num_wgt, den_k)),
        ])

    if key == "(13)":
        # r_{\mathrm{global}}(t) \exp[\mathrm{i}\Phi(t)] = \frac{1}{N} \sum_{j=1}^{N} \exp[\mathrm{i}\theta_j(t)]
        return wrap_omath_para([
            m_sub(m_run("r"), m_run("global", upright=True)),
            m_delim(m_run("t"), "(", ")"),
            m_run(" "),
            m_run("exp", upright=True),
            m_delim([m_run("i", upright=True), m_run("Φ"), m_delim(m_run("t"), "(", ")")], "[", "]"),
            m_run(" = "),
            m_frac(m_run("1", upright=True), m_run("N")),
            m_run(" "),
            m_nary(
                "∑",
                [m_run("j"), m_run("=", upright=True), m_run("1", upright=True)],
                m_run("N"),
                [m_run("exp", upright=True), m_delim([m_run("i", upright=True), m_sub(m_run("θ"), m_run("j")), m_delim(m_run("t"), "(", ")")], "[", "]")],
            ),
        ])

    if key == "(14)":
        # \mathrm{ITE}(\sigma) = \frac{\max(0, r_{\mathrm{Pearson}}[\mathrm{sgn}(x(t)), s(t)])}{1 + (\sigma / \sigma_0)^4}
        num_ite = [
            m_run("max", upright=True),
            m_delim([
                m_run("0, "),
                m_sub(m_run("r"), m_run("Pearson", upright=True)),
                m_delim([
                    m_run("sgn", upright=True),
                    m_delim([m_run("x"), m_delim(m_run("t"), "(", ")")], "(", ")"),
                    m_run(", s"),
                    m_delim(m_run("t"), "(", ")"),
                ], "[", "]"),
            ], "(", ")"),
        ]
        den_ite = [
            m_run("1 + "),
            m_sup(m_delim(m_frac(m_run("σ"), m_sub(m_run("σ"), m_run("0", upright=True))), "(", ")"), m_run("4", upright=True)),
        ]
        return wrap_omath_para([
            m_run("ITE", upright=True),
            m_delim(m_run("σ"), "(", ")"),
            m_run(" = "),
            m_frac(num_ite, den_ite),
        ])

    if key == "(15)":
        # E(\boldsymbol{\theta}) = -\frac{1}{2N} \sum_{i=1}^{N} \sum_{j=1}^{N} A_{ij} \cos(\theta_j - \theta_i)
        inner_sum = m_nary(
            "∑",
            [m_run("j"), m_run("=", upright=True), m_run("1", upright=True)],
            m_run("N"),
            [
                m_sub(m_run("A"), m_run("ij")),
                m_run(" "),
                m_run("cos", upright=True),
                m_delim([m_sub(m_run("θ"), m_run("j")), m_run(" − "), m_sub(m_run("θ"), m_run("i"))], "(", ")"),
            ],
        )
        return wrap_omath_para([
            m_run("E"),
            m_delim(m_run("θ", bold=True), "(", ")"),
            m_run(" = −"),
            m_frac(m_run("1", upright=True), [m_run("2", upright=True), m_run("N")]),
            m_run(" "),
            m_nary("∑", [m_run("i"), m_run("=", upright=True), m_run("1", upright=True)], m_run("N"), inner_sum),
        ])

    if key == "(16)":
        # \tau_{\mathrm{RMST}}(T_{\max}) = \int_{0}^{T_{\max}} \hat{S}(t) \mathrm{d}t
        s_hat = m_acc(m_run("S"), "\u0302")
        return wrap_omath_para([
            m_sub(m_run("τ"), m_run("RMST", upright=True)),
            m_delim(m_sub(m_run("T"), m_run("max", upright=True)), "(", ")"),
            m_run(" = "),
            m_nary(
                "∫",
                m_run("0", upright=True),
                m_sub(m_run("T"), m_run("max", upright=True)),
                [s_hat, m_delim(m_run("t"), "(", ")"), m_run(" "), m_run("d", upright=True), m_run("t")],
                lim_loc="subSup",
            ),
        ])

    if key == "(17)":
        # \ln \tau_{\mathrm{RMST}}(\sigma) = a + \frac{2 \Delta E_{\mathrm{eff}}}{\sigma^2}
        return wrap_omath_para([
            m_run("ln", upright=True),
            m_run(" "),
            m_sub(m_run("τ"), m_run("RMST", upright=True)),
            m_delim(m_run("σ"), "(", ")"),
            m_run(" = a + "),
            m_frac(
                [m_run("2", upright=True), m_run(" Δ"), m_sub(m_run("E"), m_run("eff", upright=True))],
                m_sup(m_run("σ"), m_run("2", upright=True)),
            ),
        ])

    # Supplementary Equations (S1) - (S4)
    if key == "(S1)":
        return wrap_omath_para([
            m_sub(m_run("A"), m_run("0", upright=True)),
            m_run(" = "),
            m_run("max", upright=True),
            m_delim([m_run("B, "), m_sup(m_run("B"), m_run("⊤", upright=True))], "(", ")"),
            m_run(" ,    "),
            m_sub(m_run("A"), m_run("0,ii")),
            m_run(" = 0 ,    ρ"),
            m_delim(m_run("t"), "(", ")"),
            m_run(" = "),
            m_frac(m_run("1", upright=True), [m_run("N"), m_delim(m_run("N − 1"), "(", ")")]),
            m_run(" "),
            m_nary(
                "∑",
                m_run("i≠j"),
                None,
                [
                    m_run("1", upright=True, bold=True),
                    m_delim([m_sub(m_run("A"), m_run("ij")), m_delim(m_run("t"), "(", ")"), m_run(" > 0")], "(", ")"),
                ],
            ),
        ])

    if key == "(S2)":
        return wrap_omath_para([
            m_sub(m_run("S"), m_run("ij")),
            m_run(" = 0.65 "),
            m_frac(m_sub(m_run("CN"), m_run("ij")), [m_run("max", upright=True), m_delim(m_run("CN"), "(", ")")]),
            m_run(" + 0.35 "),
            m_sub(m_run("W"), m_run("ij")),
            m_run(" + "),
            m_sub(m_run("ξ"), m_run("ij")),
        ])

    if key == "(S3)":
        return wrap_omath_para([
            m_sub(m_run("M"), m_run("3", upright=True)),
            m_delim(m_run("A"), "(", ")"),
            m_run(" = "),
            m_frac(m_run("1", upright=True), m_run("6", upright=True)),
            m_run(" "),
            m_run("Tr", upright=True),
            m_delim(m_sup(m_run("A"), m_run("3", upright=True)), "(", ")"),
        ])

    if key == "(S4)":
        return wrap_omath_para([
            m_sup(m_run("m"), m_run("(μ)")),
            m_run(" = "),
            m_delim([
                m_frac(m_run("1", upright=True), m_run("N")),
                m_run(" "),
                m_nary(
                    "∑",
                    [m_run("j"), m_run("=", upright=True), m_run("1", upright=True)],
                    m_run("N"),
                    [
                        m_subsup(m_run("ξ"), m_run("j"), m_run("(μ)")),
                        m_run(" "),
                        m_run("exp", upright=True),
                        m_delim([m_run("i", upright=True), m_sub(m_run("θ"), m_run("j"))], "(", ")"),
                    ],
                ),
            ], "|", "|"),
        ])

    return None
