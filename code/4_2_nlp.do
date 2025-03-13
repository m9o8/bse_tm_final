* Clear
clear

* Load data
use "../data/stata/nlp.dta"

* Make group categorical
encode group, generate(group_id)

* Set-up panel structure
xtset group_id week_index

* Run basic DiD regression
xtdidregress (mean_score) (treatment_synthdid), group(group_id) time(week_index) vce(cluster group)

********************************************* Synthetic DiD *********************************************

* Synthetic DiD 

* Quietly regress to get estimates for the plot
quiet sdid mean_score group_id week_index treatment_synthdid, vce(bootstrap) reps(100) seed(123)

* Create a more readable date format from week_start
* First, ensure week_start is formatted properly
gen plot_date = week_start
format plot_date %tdDD-Mon-YY

* Define the treatment date (when ChatGPT was introduced)
local treatment_date = "30nov2022"
 
* Get ATT and SE from model results
local att_value = e(ATT)
local se_value = e(se)
local att_rounded = round(`att_value', 0.001)
local se_rounded = round(`se_value', 0.001)

sdid mean_score group_id week_index treatment_synthdid, vce(bootstrap) reps(100) seed(123) graph g1on g1_opt(xtitle("") ///
                  title("Synthetic Control: Group Weights", size(medium)) ///
                  xlabel(#12, angle(45) labsize(small)) ///
                  legend(order(1 "Treated (Stack Overflow)" 2 "Synthetic Control")) ///
				  scheme(plotplainblind)) ///
				  g2_opt(title("Synthetic Control: Impact of ChatGPT on Question Complexity", size(medium)) ///
				  ylabel() ytitle("Complexity Score") ///
				  xtitle("Week") ///
				  xlabel(0(25)175) ///
				  text(, placement(e) color(black) size(small)) ///
				  note("ATT = `att_rounded' (SE = `se_rounded')", size(small)) ///
				  scheme(plotplainblind)) ///
				  graph_export(../imgs/stata/sdid_nlp_, .pdf)

* Synthetic DiD with covariate controls
sdid mean_score group week_index treatment_synthdid, vce(bootstrap) reps(100) covariates(year_month_*) seed(123)

local month_vars ""
foreach var of varlist year_month_* {
    local month_vars "`month_vars' `var'"
}

* Synthetic Event Study DiD
sdid_event mean_score group week_index treatment_synthdid, covariates(`month_vars') vce(bootstrap) brep(100) placebo(all)
* Extract the results matrix (adjust row numbers based on how many periods you have)
mat res = e(H)[2..165,1..5]  /* Assuming 65 post-treatment periods from your output */

* Convert matrix to variables
svmat res

* Generate relative time to treatment
gen rel_time = _n - 1 if !missing(res1)

replace rel_time = 66 - _n if _n > 66 & !missing(res1)

* Sort by relative time
sort rel_time

* Create the plot
twoway (rarea res3 res4 rel_time, lcolor(gs10) fcolor(gs11%50)) ///
       (scatter res1 rel_time, mcolor(navy) msymbol(O)) ///
       (connected res1 rel_time, lcolor(navy) lpattern(solid)), ///
       legend(order(1 "95% CI" 2 "Point Estimate") pos(6) rows(1)) ///
       title("Impact of ChatGPT on Script-Language Question Complexity", size(medium)) ///
       subtitle("Event Study: Post-Treatment Effects", size(small)) ///
       xtitle("Realtive time to ChatGPT Release (Nov 30, 2022) in weeks") ///
       ytitle("Change in Complexity Scores") ///
       yline(0, lcolor(red) lpattern(dash)) ///
       xline(0, lcolor(black) lpattern(solid)) ///
       xlabel(-100(15)65) ///
	   note("Average Treatment Effect: `att_rounded'", size(small)) ///
       scheme(s1color)
* Export the graph
graph export "../imgs/stata/event_study_nlp.pdf", replace