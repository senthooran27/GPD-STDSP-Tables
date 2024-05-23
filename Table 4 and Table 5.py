#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Table 4
import numpy as np
from IPython.display import HTML
table = np.zeros((21, 20))
row_names = [f"{i + 1}" for i in range(0,21)]
column_names = [f"{i + 1}" for i in range(0,20)]

from math import exp
import pandas as pd
p= float(input("enter value of pd: "))
alpha=float(input("enter value of dispersion(lambda 2):"))
for n1 in range(0,21,1):
    n11=int(row_names[n1])
    for n2 in range(0,20,1):
        n22=int(column_names[n2])
        if(n22<=n11)or(n22>n11) :
            a=(n11+n22)*exp(-(n11+n22)*p)*((1/(n11+n22))+(n22/(n11+n22))*p*exp(-alpha))
            d=n1-1
            f=n2-1
            table[n1,n2]=round(a, 4) #a
            
df=pd.DataFrame(table)

df.columns=column_names
df.index=row_names
df.replace(0, "--", inplace=True)
df = df.iloc[:-1]


styled_df = df.style.set_table_styles([
    {"selector": "th", "props": [("background-color", "lightgray")]},
    {"selector": "td", "props": [("text-align", "right")]},
    # Add borders to all cells
    {"selector": "th, td", "props": [("border", "1px solid black")]},
])
html_string = styled_df.to_html()

with open('df_with_scroll.html', 'w') as f:
    f.write(html_string)

import webbrowser
webbrowser.open('df_with_scroll.html')  
HTML('<iframe src="df_with_scroll.html" style="width:100%; height:400px;"></iframe>')


# In[12]:


#Table 5
# P value for double sampling
a=(n1+n2)*exp(-(n1+n2)*p)*((1/(n1+n2))+(n2/(n1+n2))*p*exp(-alpha))

import numpy as np

def solve_for_p(a, n1, n2, alpha):
    def f(p):
        return a - ((n1 + n2) * np.exp(-(n1 + n2) * p) * ((1 / (n1 + n2)) + (n2 / (n1 + n2)) * p * np.exp(-alpha)))

    try:
        from scipy.optimize import root_scalar
        result = root_scalar(f, bracket=[0, 1])  # Assume p is within [0, 1]
        return result.root
    except ImportError:
        print("SciPy is not installed. Using a less efficient method.")

        p_lower = 0
        p_upper = 1
        while abs(p_upper - p_lower) > 1e-6:
            p_mid = (p_lower + p_upper) / 2
            if f(p_mid) == 0:
                return p_mid
            elif f(p_mid) * f(p_lower) < 0:
                p_upper = p_mid
            else:
                p_lower = p_mid

        return (p_upper + p_lower) / 2

a = [0.005,0.010,0.025,0.050,0.100,0.250,0.750,0.900,0.950,0.975,0.990,0.995]
n1=float(input("sample 1 size:"))
n2 = float(input("sample 2 size:"))
alpha = float(input("dispersion value:"))

table2 = np.zeros((1,len(a)))

from math import exp
import pandas as pd
for b in range(0,len(a)):
    p = solve_for_p(a[b], n1, n2, alpha)
    table2[0,b] = round(p,4)
   

df=pd.DataFrame(table2,columns=a)
df.index=['p_values']
df

