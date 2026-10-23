"""Create the platform upload, readable report and reproducible delivery ZIP."""
from pathlib import Path
import json, zipfile, html
import pandas as pd
import numpy as np
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_LEFT
B=Path(__file__).resolve().parent; O=B/'outputs'
a=json.loads((O/'data_audit.json').read_text()); m=json.loads((O/'run_manifest.json').read_text())
f=pd.read_csv(O/'future_daily_forecast.csv'); u=f[['sku_id','date','pred_mean','pred_std']].rename(columns={'pred_mean':'mean_units','pred_std':'sd_units'})
print("Total rows:", len(u))
print("Unique SKUs:", u.sku_id.nunique())
print("Unique dates:", u.date.nunique())
assert len(u)==224000 and u.sku_id.nunique()==2000 and u.date.nunique()==112
assert u.date.min()=='2027-09-07' and u.date.max()=='2027-12-27'
assert not u.duplicated(['sku_id','date']).any() and np.isfinite(u[['mean_units','sd_units']]).all().all() and (u[['mean_units','sd_units']]>=0).all().all()
u.to_csv(O/'demand_estimates.csv',index=False,float_format='%.9g')
mc=pd.read_csv(O/'model_comparison.csv'); weekday=pd.read_csv(O/'weekday_profile.csv'); zones=pd.read_csv(O/'zone_combinations.csv'); pairs=pd.read_csv(O/'significant_sku_pairs.csv'); ots=pd.read_csv(O/'ots_budget_summary.csv'); ots_req=pd.read_csv(O/'ots_scenario_requirements.csv')
test=mc[(mc.stage=='test')&mc.selected_on_validation].iloc[0]
styles=getSampleStyleSheet();styles['Normal'].fontSize=10;styles['Normal'].leading=14;styles['Normal'].spaceAfter=8
styles['Title'].textColor=colors.HexColor('#15364b');styles['Heading1'].textColor=colors.HexColor('#087e8b');styles['Heading2'].textColor=colors.HexColor('#15364b')
story=[]; md=[]
def title(t):story.append(Paragraph(html.escape(t),styles['Title']));md.append('# '+t+'\n')
def h(t):story.append(Paragraph(html.escape(t),styles['Heading1']));md.append('## '+t+'\n')
def p(t):story.append(Paragraph(html.escape(t),styles['Normal']));md.append(t+'\n')
def table(headers,rows,widths=None):
    vals=[[Paragraph(html.escape(str(v)),styles['Normal']) for v in headers]]+[[Paragraph(html.escape(str(v)),styles['Normal']) for v in row] for row in rows]
    t=Table(vals,colWidths=widths,repeatRows=1,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e3edf1')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.8,colors.HexColor('#087e8b')),('BOTTOMPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5)]));story.extend([t,Spacer(1,10)])
    md.append('| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows)+'\n')
def pic(name,width=490,height=None):
    from PIL import Image as PILImage
    im=PILImage.open(O/name);height=height or width*im.height/im.width;story.append(Image(str(O/name),width=width,height=height));md.append(f'![{name}]({name})\n')
def page():story.append(PageBreak())
title('HyperBasket Quick Fulfiller\nTask 1 | Demand characterization')
p('Casework 2, Fall 2026 | Revised package: 2026-10-12 | Analysis: 2026-10-15')
h('Scope and principal findings')
p('This report covers Task 1(a-f). All estimates are rebuilt from the revised raw orders, order lines and SKU master. The previous daily-demand export is used only to quantify the revision; it is never used to fit the new forecasts.')
table(['Measure','Recomputed result'],[['Historical period','2026-09-07 to 2027-09-06 (365 days)'],['Products / orders / order lines',f'2,000 / {a["orders"]:,} / {a["order_lines"]:,}'],['Units / average orders per day',f'{a["units"]:,} / {a["mean_orders_daily"]:.2f}'],['Peak day',f'{a["peak_date"]}: 600 orders'],['Peak recurring period','Saturday, 17:00-18:00; 75.31 orders/hour'],['Average basket / multi-zone share',f'{a["mean_basket_units"]:.3f} units / {a["multi_zone_share"]:.2%}'],['Selected daily-demand model','B: weekday effects + log-linear trend'],['Platform upload','224,000 rows; 2,000 SKUs x 112 evaluation dates']],[180,310])
h('What changed and why the old results cannot be reused')
p('The dictionary states that orders were thinned to 34.7% of the original release, timestamps were repaired, order IDs were regenerated and opening inventory was rescaled. Our old daily export totals 1,523,721 units versus 529,134 now (34.726% retained). Uniformly scaling old standard deviations or quantiles is not statistically valid; the SKU-day panel and all order-level analyses were rebuilt.')
p('All nine supplied SHA-256 checksums match. Checks also confirm unique keys, valid joins, correct temperature zones, positive integer line quantities, basket reconciliation, consistent day/date/timestamp fields and orders inside 10:00-18:00. Missing SKU-day combinations represent zero recorded demand after these completeness checks, not missing files.')
p('Timing definitions are kept separate: 20/30/45 minutes are OTD delivery promises used to derive order-specific available OTS budgets, while Fast/Medium/Slow are facility OTS scenarios requiring 4-5, 5-7 and 8-10 minutes respectively. These are different quantities, not conflicting definitions.')
page();h('1(a) Daily product demand: estimation and validation')
p('For each product and calendar date, sum order-line quantities after joining order_id to placement_date. Complete the 2,000 x 365 panel with zeros. Historical means, sample standard deviations, empirical quantiles, zero-demand shares and monthly daily means are retained as descriptive evidence, not as future forecasts.')
p('Three candidate mean models are fitted separately to every SKU: A uses the last 56 training days grouped by weekday; B uses a Poisson GLM with log mean = intercept + six weekday indicators + linear time in years; C adds annual sine and cosine terms to B. C explicitly tests annual seasonality. With only one year, trend and seasonality are weakly identifiable.')
p('Fit days 1-253 and validate days 254-309; select a common model family by average Poisson deviance, choosing the simpler family if within 1% of the best. Freeze selection, refit on days 1-309 and test days 310-365. Finally refit the selected family on all 365 historical days and forecast the required 112 days. No evaluation-year demand is used.')
table(['Stage / model','Deviance','MAE','WAPE'],[[r.stage+' / '+r.model[0],f'{r.mean_poisson_deviance:.4f}',f'{r.mae:.4f}',f'{r.wape:.2%}'] for r in mc.itertuples()],[190,100,100,100])
p('B wins validation (0.7691 versus 2.0857 for A and 1.0000 for C). Its independent-test WAPE is 61.65%, and its test mean bias is -0.0069 units per SKU-day. High percentage error is partly driven by sparse low-volume products; it is a real forecast limitation. No claim is made that annual seasonality is absent: the annual seasonal candidate simply fails to improve this holdout comparison.')
p('Daily uncertainty uses a plug-in NB2 distribution: variance = mu + alpha x mu squared. Alpha is estimated from training residuals with an approximate degrees-of-freedom correction and truncated at zero; alpha near zero gives Poisson. Standard deviation and discrete p50/p90/p99 vary with the daily mean. This is not a joint maximum-likelihood negative-binomial fit.')
p(f'Test marginal coverage: p50 {test.coverage_p50:.2%}, p90 {test.coverage_p90:.2%}, p99 {test.coverage_p99:.2%}. Discreteness creates overcoverage, especially at zero; these are not confidence intervals for the mean or inventory fill rates. All 14,000 fits reported status ok. Parameter uncertainty and temporal/cross-SKU dependence are not included.')
page();h('1(a) Forecast evidence and handoff')
pic('test_examples.png',490)
p('Examples were selected by training-period demand rank, not test accuracy. The full per-SKU error table and fitted coefficients are included. Variability is conditional on the fitted mean; a constant annual coefficient of variation is not imposed.')
p('The 4-week versus 8-week weekday baseline sensitivity produces test WAPE 63.40% versus 61.85%. This diagnostic does not change the frozen model choice. The 56-day test is shorter than the 112-day forecast window, and only one historical annual cycle is available.')
p('demand_estimates.csv contains exactly sku_id,date,mean_units,sd_units for 2027-09-07 through 2027-12-27. The richer future_daily_forecast.csv also retains p50/p90/p99 and dispersion. Task 2 must model cumulative demand over actual supplier calendars; adding daily p90s or p99s is incorrect. Fit parameters and per-SKU residual lag-1 diagnostics are supplied for that work.')
page();h('1(b-c) Operating peaks and basket sizes')
p('Weekday averages divide by the actual number of observed days (53 Mondays, 52 of each other weekday). Hour profiles use all date-hour cells, including zero-order cells. Probability of an order occurring on a weekday is distinct from expected orders per occurrence of that weekday.')
table(['Day','Days','Mean orders/day','Order share'],[[['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][int(r.weekday)],str(r.days),f'{r.mean_orders:.2f}',f'{r.probability:.2%}'] for r in weekday.itertuples()],[100,80,160,150])
p('The largest observed day is 2026-12-26 (600 orders). The largest observed one-hour bucket is 2026-12-19, 17:00-18:00 (108 orders), versus 75.31 for the typical Saturday 17:00 hour. Staffing based only on the overall average of 45.34 orders per open hour would hide this peak. These are demand rates, not computed staffing requirements.')
p(f'Basket size means units per order: mean {a["mean_basket_units"]:.3f}, p50 = 3, p90 = 7, p99 = 12. Distinct products average {a["mean_distinct_skus"]:.3f} per order. Both complete probability distributions are supplied because picker work has both per-line and per-unit components.')
pic('operations_dashboard.png',490)
page();h('1(d-e) Co-ordering and temperature-zone combinations')
p('Represent each basket as binary SKU presence. For each unordered SKU pair, calculate support = joint orders / all orders, directional confidence = joint orders / orders containing the conditioning SKU, and lift = observed joint probability / product of marginal probabilities. A unit quantity greater than one does not create repeated co-occurrence.')
p(f'One-sided hypergeometric tests evaluate positive association conditional on SKU frequencies. Benjamini-Hochberg adjustment covers all {a["total_pair_tests"]:,} possible pairs, including unseen pairs with p = 1. There are {a["significant_pairs"]:,} pairs with adjusted q <= 0.05 and lift > 1. These are screening results under the exchangeable-order model; shared seasonality, basket size and dependent pair tests can affect interpretation. They do not establish causality or guarantee benefit from adjacent slots.')
table(['Pair','Joint orders','Support','Lift'],[[r.sku_a+' / '+r.sku_b,f'{r.co_orders:,}',f'{r.support:.2%}',f'{r.lift:.3f}'] for r in pairs.head(6).itertuples()],[190,105,100,95])
p('The highest-volume significant pair P0001/P0002 appears in 11,898 baskets (8.99%), but lift is only 1.030: statistical significance does not imply a large association. Its directional confidence is 22.40% for P0002 given P0001 and 41.31% in the reverse direction. Use frequency to prioritize travel savings and lift to interpret association, while respecting temperature and storage constraints. Full pair metrics and a separate top-100 volume ranking are included.')
table(['Zone combination','Orders','Probability'],[[r.zone_combination,f'{r.orders:,}',f'{r.probability:.2%}'] for r in zones.itertuples()],[280,105,105])
p('54.36% of baskets span at least two zones, and 9.72% require all three. Cross-zone travel and temperature-protected consolidation are routine work, not exceptions. A zone count describes whether a zone is visited; it is not the number of units or the required staging capacity. Later tasks must combine these frequencies with dwell times and peak arrivals.')
page();h('1(f) OTS requirements and OTD-derived time budgets')
p('Under calculation rule 5 and the parameter CSV, available OTS budget for order o is T - estimated_delivery_minutes(o), where T is 20, 30 or 45 minutes. Latest staged ready time is placement plus this budget. Courier arrival offset affects handoff/staging occupancy; it is not subtracted again from the deadline. The same order stream is held fixed across promises.')
table(['OTD promise','Mean OTS','p10','p50','p90','Negative'],[[str(int(r.otd_promise_minutes)),f'{r.mean:.2f}',f'{r.p10:.2f}',f'{r.p50:.2f}',f'{r.p90:.2f}',f'{r.negative_share:.1%}'] for r in ots.itertuples()],[100,90,70,70,80,80])
pic('ots_distribution.png',490)
p('These are distributions of available time, not measured picker completion times. The data have no realized OTS timestamps, so achieved service cannot be inferred. All three budgets have standard deviation 2.584 minutes and differ only by their promise offsets.')
p('Separately, the case narrative defines the facility OTS scenarios directly: Fast requires 4-5 minutes, Medium 5-7 minutes, and Slow 8-10 minutes. These are exogenous order-to-staging performance ranges for later facility experiments; they are not OTD promises and should not be subtracted from estimated delivery time.')
p('Accordingly, the 20/30/45-minute analysis above answers a different question: how much order-to-staging time is available after accounting for estimated delivery under each delivery promise. The Fast/Medium/Slow OTS ranges are retained separately in ots_scenario_requirements.csv for later capacity and service analysis.')
page();h('Reproducibility, limitations and source register')
p('Run analyze_task1.py with --source pointing to the revised package; run forecast_task1a.py using outputs/daily_demand.csv and --horizon 112; run report_task1.py to export the platform upload and this report. README.md gives exact commands. The original source files and previous-version results are preserved.')
p('Key outputs: demand_estimates.csv (platform); future_daily_forecast.csv and model_parameters_and_diagnostics.csv (Task 2 modeling); weekday_hour_profile.csv, basket distributions, significant_sku_pairs.csv and zone_combinations.csv (facility design); order_staging_deadlines.csv (rule-based time budgets). Intermediate raw daily demand and independent-test predictions are retained for audit.')
p('The original sources are identified by package filename and SHA-256 in data_audit.json. Forecast dates, model selection rule, split dates, software versions and fit status are in run_manifest.json. upload_validation.json verifies the delivered upload after reopening it. The delivery ZIP includes the source code, reports, machine-readable results and figures, but not the original package.')
h('Sources used')
p('1. ISyE 6202 & 6335 HyperBasket Quick Fulfiller Casework 2 Fall 2026 BM-PM 2026-10-12_v2.0.pdf: Task 1 on p8; Order Preparation Process defines Fast/Medium/Slow as required OTS ranges of 4-5, 5-7 and 8-10 minutes; modeling rule 5 defines the separate 20/30/45-minute OTD promises.')
p('2. QF_Facility_Data_Dictionary.xlsx: Part 1 uploads and Changes 2026-10-12 worksheets; field meanings on Orders, Order Lines and SKU Master worksheets.')
p('3. QF_Facility_Orders.csv; QF_Facility_Order_Lines.csv; QF_Facility_SKU_Master.csv; QF_Facility_Design_Parameters.csv; PACKAGE_SHA256SUMS.txt, revised data package.')
h('AI assistance and review boundary')
p('AI assisted with code adaptation, computation, data checks, visualizations and drafting. Automated checks and holdout results are documented; this report does not assert that team members have reviewed or independently verified it. Team members should inspect the supplied code and evidence, resolve the timing inconsistency with the instructor, and write an accurate contribution and AI-use statement based on their actual review.')
p('This delivery completes Task 1 analysis. It does not simulate Task 2 stock policies, infer an optimal building layout, or claim measured operational service. Single-year seasonality, rare-SKU uncertainty, unmodeled dependencies and ambiguous timing scenarios remain material limitations.')
(O/'Task1_Report_EN.md').write_text('\n'.join(md),encoding='utf-8')
def footer(canvas,doc):
    canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#526276'));canvas.drawString(42,26,'HyperBasket | Task 1 | Revised 2026-10-12 data');canvas.drawRightString(553,26,str(doc.page))
SimpleDocTemplate(str(O/'Task1_Report_EN.pdf'),pagesize=(595,842),rightMargin=42,leftMargin=42,topMargin=38,bottomMargin=42).build(story,onFirstPage=footer,onLaterPages=footer)
r=pd.read_csv(O/'demand_estimates.csv');assert r.columns.tolist()==['sku_id','date','mean_units','sd_units'];assert len(r)==224000;assert r.groupby('sku_id').size().eq(112).all();assert np.allclose(r.mean_units,f.pred_mean) and np.allclose(r.sd_units,f.pred_std)
(O/'upload_validation.json').write_text(json.dumps({'rows':len(r),'skus':r.sku_id.nunique(),'dates':r.date.nunique(),'start':r.date.min(),'end':r.date.max(),'unique_sku_date':not r.duplicated(['sku_id','date']).any(),'nonnegative_finite':bool(np.isfinite(r[['mean_units','sd_units']]).all().all() and (r[['mean_units','sd_units']]>=0).all().all()),'matches_forecast':True},indent=2))
print('Report and upload saved.')
