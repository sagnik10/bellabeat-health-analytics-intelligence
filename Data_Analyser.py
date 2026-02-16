import os
import glob
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import warnings
warnings.filterwarnings("ignore")

start_total=time.time()

BASE_DIR=os.getcwd()
INPUT_DIR=os.path.join(BASE_DIR,"Input_Data")
OUTPUT_DIR=os.path.join(BASE_DIR,"Output")
CHART_DIR=os.path.join(OUTPUT_DIR,"charts")
DATA_DIR=os.path.join(OUTPUT_DIR,"data")
MODEL_DIR=os.path.join(OUTPUT_DIR,"models")

os.makedirs(CHART_DIR,exist_ok=True)
os.makedirs(DATA_DIR,exist_ok=True)
os.makedirs(MODEL_DIR,exist_ok=True)

plt.style.use("dark_background")

DARK_BG="#0e1117"
GRID_COLOR="#2a2f3a"
TEXT_COLOR="#e6edf3"
ACCENT="#00d4ff"

UNIT_MAP={
"steps":"Steps (count)",
"totalsteps":"Steps (count)",
"calories":"Calories (kcal)",
"totalcalories":"Calories (kcal)",
"distance":"Distance (km)",
"veryactiveminutes":"Minutes",
"fairlyactiveminutes":"Minutes",
"lightlyactiveminutes":"Minutes",
"sedentaryminutes":"Minutes",
"heartrate":"Heart Rate (bpm)",
"mets":"METs"
}

files=glob.glob(os.path.join(INPUT_DIR,"*.csv"))

def process_file(file):
    chunks=[]
    for chunk in pd.read_csv(file,chunksize=300000):
        chunk=chunk.drop_duplicates()
        chunk.columns=[c.lower() for c in chunk.columns]
        if "activitydate" in chunk.columns:
            chunk["activitydate"]=pd.to_datetime(chunk["activitydate"],errors="coerce")
            chunk=chunk.groupby("activitydate").mean(numeric_only=True)
        elif "activityhour" in chunk.columns:
            chunk["activityhour"]=pd.to_datetime(chunk["activityhour"],errors="coerce")
            chunk=chunk.groupby("activityhour").mean(numeric_only=True)
        elif "time" in chunk.columns:
            chunk["time"]=pd.to_datetime(chunk["time"],errors="coerce")
            chunk=chunk.groupby(chunk["time"].dt.date).mean(numeric_only=True)
        else:
            chunk=chunk.mean(numeric_only=True).to_frame().T
        chunks.append(chunk)
    return pd.concat(chunks)

print("\nProcessing datasets")

aggregates=[]
for file in tqdm(files):
    aggregates.append(process_file(file))

combined=pd.concat(aggregates)
combined.to_excel(os.path.join(DATA_DIR,"master_aggregated.xlsx"))

numeric_cols=combined.select_dtypes(include=np.number).columns

def clip_series(s):
    lower=s.quantile(0.01)
    upper=s.quantile(0.99)
    return s.clip(lower,upper)

def apply_axis_style(ax,xlabel,ylabel,title):

    ax.set_title(title,fontsize=18,color=TEXT_COLOR,pad=15,fontweight="bold")
    ax.set_xlabel(xlabel,fontsize=13,color=TEXT_COLOR,labelpad=10)
    ax.set_ylabel(ylabel,fontsize=13,color=TEXT_COLOR,labelpad=10)
    ax.tick_params(axis='both',colors=TEXT_COLOR,labelsize=10)
    ax.grid(True,color=GRID_COLOR,alpha=0.4)

    xt=ax.get_xticks()
    yt=ax.get_yticks()

    if len(xt)>8:
        ax.set_xticks(xt[::max(1,len(xt)//6)])

    if len(yt)>8:
        ax.set_yticks(yt[::max(1,len(yt)//6)])

def save_fig(fig,name):
    plt.tight_layout()
    fig.savefig(os.path.join(CHART_DIR,name),dpi=300,facecolor=fig.get_facecolor())
    plt.close()

print("\nGenerating correlation matrices")

corr=combined[numeric_cols].corr()

fig,ax=plt.subplots(figsize=(14,10))
fig.patch.set_facecolor(DARK_BG)
sns.heatmap(corr,cmap="coolwarm",ax=ax)
apply_axis_style(ax,"Metrics","Metrics","Pearson Correlation Matrix")
save_fig(fig,"corr_pearson.png")

fig,ax=plt.subplots(figsize=(14,10))
fig.patch.set_facecolor(DARK_BG)
sns.heatmap(combined[numeric_cols].corr(method="spearman"),cmap="coolwarm",ax=ax)
apply_axis_style(ax,"Metrics","Metrics","Spearman Correlation Matrix")
save_fig(fig,"corr_spearman.png")

fig,ax=plt.subplots(figsize=(14,10))
fig.patch.set_facecolor(DARK_BG)
sns.heatmap(combined[numeric_cols].corr(method="kendall"),cmap="coolwarm",ax=ax)
apply_axis_style(ax,"Metrics","Metrics","Kendall Correlation Matrix")
save_fig(fig,"corr_kendall.png")

cov=combined[numeric_cols].cov()

fig,ax=plt.subplots(figsize=(14,10))
fig.patch.set_facecolor(DARK_BG)
sns.heatmap(cov,cmap="coolwarm",ax=ax)
apply_axis_style(ax,"Metrics","Metrics","Covariance Matrix")
save_fig(fig,"covariance_matrix.png")

print("\nGenerating 20 chart types")

for col in tqdm(numeric_cols[:5]):

    s=clip_series(combined[col].dropna())
    unit=UNIT_MAP.get(col.lower(),col)

    fig,ax=plt.subplots(figsize=(14,7))
    fig.patch.set_facecolor(DARK_BG)
    ax.plot(s.values,color=ACCENT)
    apply_axis_style(ax,"Observation Index",unit,f"{col.upper()} LINE ANALYSIS")
    save_fig(fig,f"{col}_line.png")

    fig,ax=plt.subplots(figsize=(14,7))
    fig.patch.set_facecolor(DARK_BG)
    ax.hist(s,bins=30,color=ACCENT)
    apply_axis_style(ax,unit,"Frequency",f"{col.upper()} HISTOGRAM ANALYSIS")
    save_fig(fig,f"{col}_hist.png")

    fig,ax=plt.subplots(figsize=(14,7))
    fig.patch.set_facecolor(DARK_BG)
    sns.boxplot(x=s,ax=ax,color=ACCENT)
    apply_axis_style(ax,unit,"Distribution",f"{col.upper()} BOX ANALYSIS")
    save_fig(fig,f"{col}_box.png")

    fig,ax=plt.subplots(figsize=(14,7))
    fig.patch.set_facecolor(DARK_BG)
    sns.kdeplot(s,ax=ax,color=ACCENT)
    apply_axis_style(ax,unit,"Density",f"{col.upper()} DENSITY ANALYSIS")
    save_fig(fig,f"{col}_density.png")

print("\nGenerating PCA")

scaled=StandardScaler().fit_transform(combined[numeric_cols].fillna(0))

pca=PCA(n_components=2)
proj=pca.fit_transform(scaled)

fig,ax=plt.subplots(figsize=(14,7))
fig.patch.set_facecolor(DARK_BG)
ax.scatter(proj[:,0],proj[:,1],color=ACCENT)
apply_axis_style(ax,"Principal Component 1","Principal Component 2","PCA Projection")
save_fig(fig,"pca_projection.png")

print("\nGenerating anomaly detection")

iso=IsolationForest(contamination=0.01)
anomaly=iso.fit_predict(scaled)

fig,ax=plt.subplots(figsize=(14,7))
fig.patch.set_facecolor(DARK_BG)
ax.scatter(range(len(anomaly)),combined[numeric_cols[0]],c=anomaly,cmap="coolwarm")
apply_axis_style(ax,"Observation Index",numeric_cols[0],"Anomaly Detection Analysis")
save_fig(fig,"anomaly_detection.png")

print("\nGenerating forecast")

target=numeric_cols[0]

X=combined[numeric_cols].fillna(0).iloc[:-1]
y=combined[target].shift(-1).dropna()

X=X.iloc[:len(y)]

model=RandomForestRegressor(n_estimators=200)
model.fit(X,y)

pred=model.predict(X)

fig,ax=plt.subplots(figsize=(14,7))
fig.patch.set_facecolor(DARK_BG)
ax.plot(clip_series(y),label="Actual",color=ACCENT)
ax.plot(clip_series(pd.Series(pred)),label="Predicted",color="orange")
ax.legend()
apply_axis_style(ax,"Observation Index",target,"Forecast Analysis")
save_fig(fig,"forecast.png")

summary={
"Total Records":len(combined),
"Metrics":len(numeric_cols),
"Images Generated":len(glob.glob(os.path.join(CHART_DIR,"*.png")))
}

print("\nGenerating executive report")

doc=SimpleDocTemplate(os.path.join(OUTPUT_DIR,"Executive_Report.pdf"))

styles=getSampleStyleSheet()

elements=[]

elements.append(Paragraph("<b>Bellabeat Executive Intelligence Report</b>",styles["Heading1"]))
elements.append(Spacer(1,20))

for k,v in summary.items():
    elements.append(Paragraph(f"{k}: {v}",styles["Normal"]))
    elements.append(Spacer(1,10))

images=glob.glob(os.path.join(CHART_DIR,"*.png"))

for img in images:
    elements.append(Image(img,width=450,height=300))
    elements.append(Spacer(1,20))

doc.build(elements)

print("\nExecution time:",round(time.time()-start_total,2),"seconds")
print("Images created:",len(images))
print("Complete")