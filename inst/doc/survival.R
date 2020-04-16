## ---- include = FALSE---------------------------------------------------------
knitr::opts_chunk$set(
  cache = FALSE,
  collapse = TRUE,
  comment = "#>"
)
set.seed(1)
lgr::get_logger("mlr3")$set_threshold("error")

## -----------------------------------------------------------------------------
library(mlr3proba); library(mlr3); library(survival)

# type = "right" is default

TaskSurv$new(id = "right_censored", backend = survival::rats,
             time = "time", event = "status", type = "right")

task = TaskSurv$new(id = "interval_censored", backend = survival::bladder2[,-c(1, 7)],
                    time = "start", time2 = "stop", type = "interval2")
task
task$truth()[1:10]

## -----------------------------------------------------------------------------

# create task and learner

veteran = mlr3misc::load_dataset("veteran", package = "survival")
task_veteran = TaskSurv$new(id = "veteran", backend = veteran, time = "time", event = "status")
learner = lrn("surv.coxph")

# train/test split 

train_set = sample(task_veteran$nrow, 0.8 * task_veteran$nrow)
test_set = setdiff(seq_len(task_veteran$nrow), train_set)

# fit Cox PH and inspect model

learner$train(task_veteran, row_ids = train_set)
learner$model

# make predictions for new data

prediction = learner$predict(task_veteran, row_ids = test_set)
prediction

## -----------------------------------------------------------------------------
# In the previous example, Cox model predicts `lp` so `crank` is identical

all(prediction$lp == prediction$crank)
prediction$lp[1:10]

# These are evaluated with measures of discrimination and calibration.
# As all PredictionSurv objects will return crank, Harrell's C is the default measure.

prediction$score()

# distr is evaluated with probabilistic scoring rules.

measure = lapply(c("surv.graf", "surv.grafSE"), msr)
prediction$score(measure)

# Often measures can be integrated over mutliple time-points, or return
# predictions for single time-points

measure = msr("surv.graf", times = 60)
prediction$score(measure)

## -----------------------------------------------------------------------------
task = tgen("simsurv")$generate(20)
learner = lrn("surv.coxph")

# In general it is not advised to train/predict on same data

prediction = learner$train(task)$predict(task)

# The predicted `distr` is a VectorDistribution consisting of 20 separate distributions

prediction$distr

# These can be extracted and queried either invidually...

prediction$distr[1]$survival(60:70)
prediction$distr[1]$mean()

# ...or together

prediction$distr$cdf(60)[,1:10]
prediction$distr$mean()[,1:10]

# As well as plotted

plot(prediction$distr[1], "survival", main = "First 2 Survival Curves")
lines(prediction$distr[2], "survival", col = 2)

## -----------------------------------------------------------------------------
library(mlr3pipelines)

# PipeOpDistrCompositor - Train one model with a baseline distribution,
# (Kaplan-Meier or Nelson-Aalen), and another with a predicted linear predictor.
task = tgen("simsurv")$generate(30)
leaner_lp = lrn("surv.gbm", bag.fraction = 1, n.trees = 50L)
leaner_distr = lrn("surv.kaplan")
prediction_lp = leaner_lp$train(task)$predict(task)
prediction_distr = leaner_distr$train(task)$predict(task)
prediction_lp$distr

# Doesn't need training. Base = baseline distribution. ph = Proportional hazards.

pod = po("distrcompose", param_vals = list(form = "ph", overwrite = FALSE))
prediction = pod$predict(list(base = prediction_distr, pred = prediction_lp))$output

# Now we have a predicted distr!

prediction$distr

# This can all be simplified by using the distrcompose wrapper

gbm.distr = distrcompositor(learner = lrn("surv.gbm", bag.fraction = 1, n.trees = 50L),
                             estimator = "kaplan",
                             form = "aft",
                             overwrite = FALSE)
gbm.distr$train(task)$predict(task)

## -----------------------------------------------------------------------------
# PipeOpCrankCompositor - Only one model required.
leaner = lrn("surv.coxph")
prediction = leaner$train(task)$predict(task)

# Doesn't need training - Note: no `overwrite` option as `crank` is always
# present so the compositor if used will always overwrite.

poc = po("crankcompose", param_vals = list(method = "mean"))
composed_prediction = poc$predict(list(prediction))$output

# Note that whilst the actual values of `lp` and `crank` are different,
# the rankings are the same, so discrimination measures are unchanged.

prediction$crank[1:10]
composed_prediction$crank[1:10]
all(order(prediction$crank) == order(composed_prediction$crank))
cbind(Original = prediction$score(), Composed = composed_prediction$score())

# Again a wrapper can be used to simplify this
crankcompositor(lrn("surv.coxph"), method = "mean")$train(task)$predict(task)

## -----------------------------------------------------------------------------
library(mlr3pipelines); library(mlr3); library(mlr3tuning); library(paradox)
set.seed(42)

task = tgen("simsurv")$generate(50)

composed_lrn_gbm = distrcompositor(lrn("surv.gbm", bag.fraction = 1, n.trees = 50L),
                                   "kaplan", "ph")

lrns = lapply(paste0("surv.", c("kaplan", "coxph")), lrn)

design = benchmark_grid(tasks = task, learners = c(lrns, list(composed_lrn_gbm)),
                        resamplings = rsmp("cv", folds = 2))

bm = benchmark(design)
bm$aggregate(lapply(c("surv.harrellC","surv.graf","surv.grafSE"), msr))[,c(4, 7:9)]

