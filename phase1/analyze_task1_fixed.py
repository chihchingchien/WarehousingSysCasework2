"""Task 1 a-f: rebuild from v2 raw data. Run this, then forecast_task1a.py, then report_task1.py."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import hypergeom
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument('--source', type=Path, default=Path('..'))
args = ap.parse_args()

src = args.source
out = BASE / 'outputs'
out.mkdir(exist_ok=True)
def save(df,name): df.to_csv(out/name,index=False,float_format='%.9g')
checks={}
for line in (src/'PACKAGE_SHA256SUMS.txt').read_text().splitlines():
    sha,name=line.split(maxsplit=1); name=name.strip(); actual=hashlib.sha256((src/name).read_bytes()).hexdigest(); checks[name]=actual==sha
assert all(checks.values()),checks
orders=pd.read_csv(src/'QF_Facility_Orders.csv',parse_dates=['placement_date','placement_timestamp'])
lines=pd.read_csv(src/'QF_Facility_Order_Lines.csv'); skus=pd.read_csv(src/'QF_Facility_SKU_Master.csv')
assert orders.order_id.is_unique and skus.sku_id.is_unique
assert not lines.duplicated(['order_id','sku_id']).any()
assert set(lines.order_id)==set(orders.order_id) and set(lines.sku_id)<=set(skus.sku_id)
assert (lines.quantity>0).all() and (lines.quantity%1==0).all()
z=lines.merge(skus[['sku_id','temperature_zone']],on='sku_id',validate='many_to_one',suffixes=('','_master'))
assert (z.temperature_zone==z.temperature_zone_master).all()
assert orders.set_index('order_id').basket_size_units.sort_index().equals(lines.groupby('order_id').quantity.sum().sort_index().rename('basket_size_units'))
assert (orders.placement_timestamp.dt.normalize()==orders.placement_date).all()
assert (orders.day==(orders.placement_date-orders.placement_date.min()).dt.days).all()
assert (orders.day_of_week==orders.placement_date.dt.dayofweek).all()
minute=orders.placement_timestamp.dt.hour*60+orders.placement_timestamp.dt.minute+orders.placement_timestamp.dt.second/60
assert np.max(np.abs(orders.placement_minute-(orders.day*1440+minute)))<.001
assert minute.between(600,1080,inclusive='left').all()
orders['hour']=orders.placement_timestamp.dt.hour
joined=lines.merge(orders[['order_id','placement_date']],on='order_id',validate='many_to_one')
dates=pd.date_range(orders.placement_date.min(),orders.placement_date.max()); ids=skus.sku_id.sort_values()
wide=joined.groupby(['placement_date','sku_id']).quantity.sum().unstack().reindex(index=dates,columns=ids).fillna(0).astype(int)
assert wide.shape==(365,2000) and wide.to_numpy().sum()==lines.quantity.sum()
save(wide.rename_axis(index='demand_date',columns='sku_id').stack().rename('demand_units').reset_index()[['sku_id','demand_date','demand_units']],'daily_demand.csv')
summary=pd.DataFrame({'mean_daily_units':wide.mean(),'sd_daily_units':wide.std(),'zero_day_share':wide.eq(0).mean(),'annual_units':wide.sum()})
for q in [.5,.9,.99]:summary[f'empirical_p{int(q*100)}']=np.quantile(wide,q,axis=0,method='inverted_cdf')
summary=summary.rename_axis('sku_id').reset_index().merge(skus[['sku_id','temperature_zone','product_family_id']],on='sku_id')
save(summary,'sku_historical_summary.csv')
save(wide.resample('MS').mean().rename_axis(index='month',columns='sku_id').stack().rename('mean_daily_units').reset_index(),'sku_monthly_mean.csv')
daily=orders.groupby('placement_date').agg(orders=('order_id','size'),units=('basket_size_units','sum')).reindex(dates,fill_value=0)
daily['weekday']=daily.index.dayofweek; save(daily.rename_axis('date').reset_index(),'daily_operations.csv')
weekday=daily.groupby('weekday').agg(days=('orders','size'),orders=('orders','sum'),mean_orders=('orders','mean'),sd_orders=('orders','std'),max_orders=('orders','max'))
weekday['probability']=weekday.orders/len(orders); save(weekday.reset_index(),'weekday_profile.csv')
idx=pd.MultiIndex.from_product([dates,range(10,18)],names=['date','hour'])
hourly=orders.groupby(['placement_date','hour']).size().rename_axis(['date','hour']).reindex(idx,fill_value=0).rename('orders').reset_index(); hourly['weekday']=hourly.date.dt.dayofweek
heat=hourly.groupby(['weekday','hour']).orders.mean().unstack(); save(hourly,'hourly_operations.csv')
save(hourly.groupby(['weekday','hour']).orders.agg(['mean','std','max']).reset_index(),'weekday_hour_profile.csv')
basket=orders[['order_id','basket_size_units']].merge(lines.groupby('order_id').size().rename('distinct_skus'),on='order_id')
for col in ['basket_size_units','distinct_skus']:
    pmf=basket[col].value_counts().sort_index().rename('orders').rename_axis(col).reset_index();pmf['probability']=pmf.orders/len(orders);pmf['cdf']=pmf.probability.cumsum();save(pmf,col+'_distribution.csv')
# Unordered SKU pairs; binary presence, quantities do not create repeated associations.
r=pd.Categorical(lines.order_id,categories=orders.order_id).codes;c=pd.Categorical(lines.sku_id,categories=ids).codes
x=csr_matrix((np.ones(len(lines),dtype=np.int64),(r,c)),shape=(len(orders),len(ids)))
co=(x.T@x).tocoo(); keep=co.row<co.col; i,j,n=co.row[keep],co.col[keep],co.data[keep]; freq=np.asarray(x.sum(axis=0)).ravel(); N=len(orders)
p=hypergeom.sf(n-1,N,freq[i],freq[j]); total_tests=len(ids)*(len(ids)-1)//2
order=np.argsort(p); adjusted=np.minimum.accumulate((p[order]*total_tests/np.arange(1,len(p)+1))[::-1])[::-1]; q=np.empty_like(p);q[order]=np.minimum(1,adjusted)
pairs=pd.DataFrame({'sku_a':ids.to_numpy()[i],'sku_b':ids.to_numpy()[j],'co_orders':n,'support':n/N,'confidence_b_given_a':n/freq[i],'confidence_a_given_b':n/freq[j],'lift':n*N/(freq[i]*freq[j]),'p_value':p,'bh_q_all_pairs':q})
pairs['significant']= (pairs.bh_q_all_pairs<=.05)&(pairs.lift>1)
pairs=pairs.sort_values(['significant','co_orders','lift'],ascending=False);save(pairs,'observed_sku_pairs.csv')
sig=pairs[pairs.significant].copy();save(sig,'significant_sku_pairs.csv')
# Report high-volume co-picks separately from significance.
save(pairs.sort_values('co_orders',ascending=False).head(100),'top_100_copick_pairs.csv')
comb=lines.groupby('order_id').temperature_zone.agg(lambda s: '+'.join(sorted(set(s))))
zones=comb.value_counts().rename('orders').rename_axis('zone_combination').reset_index(); zones['probability']=zones.orders/N;save(zones,'zone_combinations.csv')
# Rules 5-7: latest ready = placement + OTD promise - estimated delivery.
# Keep negative budgets: they reveal infeasibility, not a zero-minute requirement.
ots=[];detail=orders[['order_id','placement_timestamp','estimated_delivery_minutes','courier_arrival_offset_minutes']].copy()
for promise in [20,30,45]:
    budget=promise-orders.estimated_delivery_minutes
    detail[f'ots_budget_{promise}_min']=budget
    detail[f'latest_staged_{promise}']=orders.placement_timestamp+pd.to_timedelta(budget,unit='m')
    detail[f'courier_arrival_{promise}']=detail[f'latest_staged_{promise}']+pd.to_timedelta(orders.courier_arrival_offset_minutes,unit='m')
    ots.append({'otd_promise_minutes':promise,'mean':budget.mean(),'sd':budget.std(),'min':budget.min(),'p01':budget.quantile(.01),'p10':budget.quantile(.1),'p50':budget.median(),'p90':budget.quantile(.9),'p99':budget.quantile(.99),'max':budget.max(),'negative_share':(budget<0).mean()})
save(pd.DataFrame(ots),'ots_budget_summary.csv');save(detail,'order_staging_deadlines.csv')
# The case narrative defines Fast/Medium/Slow as REQUIRED OTS ranges,
# not OTD promises. Keep them as exogenous facility-performance scenarios.
ots_scenarios=pd.DataFrame([
    {'ots_scenario':'Fast','required_ots_min':4,'required_ots_max':5},
    {'ots_scenario':'Medium','required_ots_min':5,'required_ots_max':7},
    {'ots_scenario':'Slow','required_ots_min':8,'required_ots_max':10},
])
save(ots_scenarios,'ots_scenario_requirements.csv')
plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'font.size':10})
fig,axs=plt.subplots(2,2,figsize=(12,8));fig.subplots_adjust(hspace=.4,wspace=.3)
axs[0,0].plot(daily.index,daily.orders,color='#087e8b',lw=.7);axs[0,0].plot(daily.index,daily.orders.rolling(28).mean(),color='#c67d20');axs[0,0].set(title='Orders per day and trailing 28-day mean',ylabel='Orders');axs[0,0].tick_params(axis='x',rotation=25)
im=axs[0,1].imshow(heat,aspect='auto',cmap='YlGnBu');axs[0,1].set(xticks=range(8),xticklabels=range(10,18),yticks=range(7),yticklabels=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'],title='Mean orders per hour by weekday',xlabel='Hour starting');fig.colorbar(im,ax=axs[0,1])
b=orders.basket_size_units.value_counts(normalize=True).sort_index();axs[1,0].bar(b.index,b.values,color='#087e8b');axs[1,0].set(title='Basket size: units (not distinct SKUs)',xlabel='Units per basket',ylabel='Probability')
axs[1,1].barh(zones.zone_combination,zones.probability,color='#3b82a6');axs[1,1].set(title='Temperature-zone combinations',xlabel='Probability')
fig.savefig(out/'operations_dashboard.png',dpi=160,bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(figsize=(9,4))
for t in [20,30,45]:
    b=np.sort(t-orders.estimated_delivery_minutes.to_numpy());ax.plot(b,np.arange(1,N+1)/N,label=f'OTD {t} min')
ax.axvline(0,color='gray',ls='--');ax.set(xlabel='Available order-to-stage time (minutes)',ylabel='Empirical cumulative probability',title='Derived OTS budgets under the explicit calculation rules');ax.legend();fig.tight_layout();fig.savefig(out/'ots_distribution.png',dpi=160);plt.close(fig)
stats={'source':str(src),'checksum_verified':checks,'orders':N,'order_lines':len(lines),'units':int(lines.quantity.sum()),'mean_orders_daily':daily.orders.mean(),'min_orders_daily':int(daily.orders.min()),'max_orders_daily':int(daily.orders.max()),'peak_date':str(daily.orders.idxmax().date()),'peak_weekday':int(weekday.mean_orders.idxmax()),'peak_weekday_hour':[int(v) for v in heat.stack().idxmax()],'peak_mean_hour_orders':float(heat.stack().max()),'max_single_hour':hourly.loc[hourly.orders.idxmax()].astype(str).to_dict(),'mean_basket_units':orders.basket_size_units.mean(),'basket_unit_quantiles':orders.basket_size_units.quantile([.5,.9,.99]).to_dict(),'mean_distinct_skus':basket.distinct_skus.mean(),'multi_zone_share':float(comb.str.contains('+',regex=False).mean()),'significant_pairs':len(sig),'observed_pairs':len(pairs),'total_pair_tests':total_tests,'all_zero_skus':int((wide.sum()==0).sum()),'estimated_delivery_mean':orders.estimated_delivery_minutes.mean()}
old=Path('/Users/tiaohu/t1_daily_demand_202610140011.csv')
if old.exists():
    olddf=pd.read_csv(old,usecols=['sku_id','demand_date','demand_units']);stats['previous_daily_export_units']=float(olddf.demand_units.sum());stats['new_to_previous_units_ratio']=float(lines.quantity.sum()/olddf.demand_units.sum())
(out/'data_audit.json').write_text(json.dumps(stats,indent=2),encoding='utf-8'); print(json.dumps(stats,indent=2))
