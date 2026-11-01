"""Exact schematic of Theorem 2; not empirical observations or human data."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ROOT=Path(__file__).resolve().parents[1]
fig,ax=plt.subplots(figsize=(6.5,2.1));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
def box(x,y,w,h,title,body,color):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.01',facecolor=color,edgecolor='#607887',linewidth=.9))
 ax.text(x+.02,y+h-.09,title,fontsize=9,weight='bold',va='top')
 ax.text(x+.02,y+h-.24,body,fontsize=8.5,va='top',linespacing=1.4)
box(.02,.28,.29,.46,'Fixed evidence',r'$p=1/2$'+'\nSymmetric user stances\nAgreement pressure '+r'$a=k/\beta$','#edf3f5')
box(.43,.55,.54,.39,'Pooled one-turn audit',r'$P(Z=z,Q=q)=1/4$'+' for all four outcomes\nSame pooled law for every '+r'$a$'+', even after repair','#eff7f4')
box(.43,.04,.54,.40,'Mean-response feedback (no repair)',r'$a<2$'+': convergence to a common center\n'+r'$a>2$'+': two separated stable equilibria','#fcf2e8')
for y,label in [(.74,'pool stance'),(.23,'retain stance')]:
 ax.annotate('',xy=(.42,y),xytext=(.32,.51),arrowprops=dict(arrowstyle='->',color='#526b7a',lw=1.2))
 ax.text(.335,y+(.035 if y>.5 else -.07),label,fontsize=7.2,rotation=0,ha='center')
fig.subplots_adjust(left=0,right=1,bottom=.02,top=.99)
fig.savefig(ROOT/'figures/audit_concept.pdf');fig.savefig(ROOT/'figures/audit_concept.png',dpi=180);plt.close(fig)
