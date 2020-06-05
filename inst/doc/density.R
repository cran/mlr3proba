## ---- include = FALSE---------------------------------------------------------
knitr::opts_chunk$set(
  cache = FALSE,
  collapse = TRUE,
  comment = "#>"
)
set.seed(1)
lgr::get_logger("mlr3")$set_threshold("error")

## -----------------------------------------------------------------------------
library(mlr3proba); library(mlr3)

task = TaskDens$new(id = "mpg", backend = datasets::mtcars, target = "mpg")

task

task$truth()[1:10]

## -----------------------------------------------------------------------------
# create task and learner

task_faithful = TaskDens$new(id = "eruptions", backend = datasets::faithful,
                             target = "eruptions")
learner = lrn("dens.kde")

# train/test split 

train_set = sample(task_faithful$nrow, 0.8 * task_faithful$nrow)
test_set = setdiff(seq_len(task_faithful$nrow), train_set)

# fitting KDE and model inspection

learner$train(task_faithful, row_ids = train_set)
learner$model
class(learner$model)

# make predictions for new data

prediction = learner$predict(task_faithful, row_ids = test_set)

## -----------------------------------------------------------------------------
prediction

# `pdf` is evaluated using the `log-loss`

prediction$score()

