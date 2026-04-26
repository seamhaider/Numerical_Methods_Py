import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 🔒 SymPy
from sympy import symbols, sympify, lambdify, sin, cos, tan, log, exp

# ---------- Page ----------
st.set_page_config(page_title="Numerical Integration Tool", layout="centered")

# ---------- CSS ----------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #1e293b, #334155);
}
.card {
    background: rgba(255,255,255,0.08);
    padding: 22px;
    border-radius: 18px;
    margin-bottom: 20px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.3);
}
.title {
    text-align:center;
    font-size:38px;
    color:#38bdf8;
    font-weight:bold;
}
.subtitle {
    text-align:center;
    color:#cbd5f5;
    margin-bottom:20px;
}
.stButton>button {
    width:100%;
    border-radius:12px;
    background: linear-gradient(135deg,#22c55e,#4ade80);
    color:white;
}
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown('<div class="title">🔢 Numerical Integration Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Simpson’s 3/8 Rule & Weddle’s Rule</div>', unsafe_allow_html=True)

# ---------- Function ----------
def f(x, eq):
    try:
        eq = eq.strip()
        eq = eq.replace("^", "**")
        eq = eq.replace("sinx", "sin(x)")
        eq = eq.replace("cosx", "cos(x)")
        eq = eq.replace("tanx", "tan(x)")
        eq = eq.replace("logx", "log(x)")

        x_sym = symbols('x')

        expr = sympify(eq, locals={
            "sin": sin,
            "cos": cos,
            "tan": tan,
            "log": log,
            "exp": exp
        })

        func = lambdify(x_sym, expr, "numpy")
        return func(x)

    except:
        return None

# ---------- Simpson ----------
def simpson38(eq, a, b, n):
    if n % 3 != 0:
        st.error("❌ n must be multiple of 3")
        return None

    h = (b - a) / n
    x_vals, f_vals, coeffs = [], [], []
    total = 0

    for i in range(n+1):
        x = a + i*h
        fx = f(x, eq)
        if fx is None:
            st.error("Invalid function input")
            return None

        if i==0 or i==n:
            c=1
        elif i%3==0:
            c=2
        else:
            c=3

        total += c*fx
        x_vals.append(x)
        f_vals.append(fx)
        coeffs.append(c)

    return (3*h/8)*total, x_vals, f_vals, coeffs, h

# ---------- Weddle ----------
def weddle(eq, a, b, n):
    if n % 6 != 0:
        st.error("❌ n must be multiple of 6")
        return None

    h = (b - a) / n
    total = 0
    x_vals, f_vals, coeffs = [], [], []

    for i in range(n+1):
        x = a + i*h
        fx = f(x, eq)
        if fx is None:
            st.error("Invalid function input")
            return None

        mod = i % 6
        if mod==0: c=1
        elif mod==1: c=5
        elif mod==2: c=1
        elif mod==3: c=6
        elif mod==4: c=1
        elif mod==5: c=5

        total += c*fx
        x_vals.append(x)
        f_vals.append(fx)
        coeffs.append(c)

    return (3*h/10)*total, x_vals, f_vals, coeffs, h

# ---------- Input ----------
st.markdown('<div class="card">', unsafe_allow_html=True)

eq = st.text_input("Enter function (e.g. sin(x)+x**3-log(x))")
st.caption("You can also write: sinx + x^3 - logx")

a = st.number_input("Lower limit (a)", value=0.0)
b = st.number_input("Upper limit (b)", value=1.0)
n = st.number_input("Number of intervals (n)", step=1, value=6)

method = st.selectbox("Select Method", ["Simpson 3/8","Weddle"])
calc = st.button("🚀 Calculate")

st.markdown('</div>', unsafe_allow_html=True)

# ---------- Calculation ----------
if calc:
    if a>=b:
        st.error("❌ a must be less than b")
    else:
        res = simpson38(eq,a,b,int(n)) if method=="Simpson 3/8" else weddle(eq,a,b,int(n))

        if res:
            result,x_vals,f_vals,coeffs,h = res

            # Result
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.success(f"Result = {result:.6f}")
            st.info(f"Step size h = {h:.6f}")
            st.markdown('</div>', unsafe_allow_html=True)

            # Table
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("📊 Iteration Table")
            df = pd.DataFrame({
                "i":range(len(x_vals)),
                "x":x_vals,
                "f(x)":f_vals,
                "Coefficient":coeffs
            })
            st.dataframe(df)
            st.markdown('</div>', unsafe_allow_html=True)

            # Graph
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("📈 Function Graph")
            x_plot = np.linspace(a,b,200)
            y_plot = [f(x,eq) for x in x_plot]

            fig,ax = plt.subplots(figsize=(4,3))
            ax.plot(x_plot,y_plot)
            ax.scatter(x_vals,f_vals)
            ax.grid(True)
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)

# ---------- Theory ----------
st.markdown('<div class="card">', unsafe_allow_html=True)

st.header("📚 Theory & Background")

st.write("""
Numerical integration is used when a definite integral cannot be solved analytically 
or when the function is known only at discrete points. The Newton–Cotes formulas approximate 
integrals by fitting polynomials through equally spaced data points.
""")

st.markdown("""
- Trapezoidal Rule → Linear approximation  
- Simpson’s 1/3 Rule → Quadratic approximation  
- Simpson’s 3/8 Rule → Cubic approximation  
- Weddle’s Rule → 6th-degree polynomial  
""")

st.subheader("🔹 Simpson’s 3/8 Rule")

st.latex(r"\int_a^b f(x)\,dx \approx \frac{3h}{8}[f(x_0)+3f(x_1)+3f(x_2)+f(x_3)]")

st.write("""
- Uses cubic polynomial  
- Requires number of intervals to be multiple of 3  
- More accurate than trapezoidal rule  
- Error term: O(h⁵)  
""")

st.subheader("🔹 Weddle’s Rule")

st.latex(r"\int_a^b f(x)\,dx \approx \frac{3h}{10}[f_0+5f_1+f_2+6f_3+f_4+5f_5+f_6]")

st.write("""
- Uses 6th-degree polynomial  
- Requires intervals multiple of 6  
- Very high accuracy  
- Error term: O(h⁷)  
""")

# Comparison
st.subheader("📊 Comparison")

comp = pd.DataFrame({
    "Feature":["Polynomial Degree","Accuracy","Condition","Error"],
    "Simpson 3/8":["3rd","Moderate","n % 3 = 0","O(h⁵)"],
    "Weddle":["6th","High","n % 6 = 0","O(h⁷)"]
})

st.table(comp)

st.markdown('</div>', unsafe_allow_html=True)

# ---------- Team ----------
st.markdown('<div class="card">', unsafe_allow_html=True)

st.header("👨‍💻 Project Team")

team = pd.DataFrame({
    "Member Name":[
        "Ekbal Haider Seam",
        "Joyjit Paul Ayan",
        "Md. Fahim Muntasir Dip",
        "Tuba Chowdhury",
        "Sumya Supty"
    ],
    "Student ID":[
        "241-15-930",
        "241-15-654",
        "241-15-812",
        "241-15-693",
        "241-15-448"
    ]
})

st.table(team)

st.markdown('</div>', unsafe_allow_html=True)
