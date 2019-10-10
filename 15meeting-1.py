#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_excel(r"D:\3333.xlsx")
df.head(10)
df.head().T


# In[2]:


df.dtypes


# In[3]:


df.isna().sum()


# In[4]:


df.dropna(inplace=True)
df.describe().T


# In[ ]:





# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import warnings


# In[6]:


corr = df.corr().round(2)


# In[7]:


plt.figure(figsize=(3,3))


# In[8]:


fig = plt.figure()


# In[9]:


sns.heatmap(corr, linewidths=0, square=True, annot=True, cmap='RdYlBu_r')


# In[10]:


plt.savefig('xx.png')


# In[11]:


import pandas as pd
df = pd.read_excel(r"D:\5555.xlsx")
df.head(10)
df.head().T


# In[12]:


df.dtypes


# In[13]:


df.isna().sum()


# In[14]:


df.dropna(inplace=True)
df.describe().T


# In[15]:


corr = df.corr().round(2)


# In[16]:


plt.figure(figsize=(3,3))


# In[17]:


fig = plt.figure()


# In[18]:


sns.heatmap(corr, linewidths=0, square=True, annot=True, cmap='RdYlBu_r')


# In[19]:


import pandas as pd
df = pd.read_excel(r"D:\6666.xlsx")
df.head(10)
df.head().T


# In[20]:


from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


# In[21]:


fig = plt.figure()
ax = Axes3D(fig)
X = np.arange(-4, 4, 0.25)
Y = np.arange(-4, 4, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')

plt.show()


# In[1]:


import pandas as pd
df = pd.read_excel(r"D:\9999.xlsx")
df.head(10)
df.head().T


# In[2]:


df.dtypes


# In[3]:


df.isna().sum()


# In[4]:


df.dropna(inplace=True)
df.describe().T


# In[14]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import warnings


# In[15]:


orr = df.corr().round(2)


# In[16]:


plt.figure(figsize=(3,3))


# In[17]:


sns.heatmap(corr, linewidths=0, square=True, annot=True, cmap='RdYlBu_r')


# In[18]:


import pandas as pd
df = pd.read_excel(r"D:\8888.xlsx")
df.head(10)
df.head().T


# In[19]:


df.dtypes


# In[20]:


df.isna().sum()


# In[21]:


df.dropna(inplace=True)
df.describe().T


# In[22]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import warnings


# In[23]:


orr = df.corr().round(2)


# In[24]:


plt.figure(figsize=(3,3))


# In[25]:


sns.heatmap(corr, linewidths=0, square=True, annot=True, cmap='RdYlBu_r')


# In[ ]:




