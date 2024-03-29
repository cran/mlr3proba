task = tsk("rats")$filter(sample(300, 50))
task_regr = tgen("friedman1")$generate(20)

test_that("crankcompositor", {
  pipe = mlr3pipelines::ppl("crankcompositor", learner = lrn("surv.kaplan"))
  expect_class(pipe, "Graph")
  pipe = mlr3pipelines::ppl("crankcompositor", learner = lrn("surv.kaplan"), graph_learner = TRUE)
  expect_class(pipe, "GraphLearner")
  pipe$train(task)
  p = pipe$predict(task)
  expect_prediction_surv(p)
  expect_true("crank" %in% p$predict_types)
})

test_that("distrcompositor", {
  pipe = mlr3pipelines::ppl("distrcompositor", learner = lrn("surv.rpart"))
  expect_class(pipe, "Graph")

  pipe = mlr3pipelines::ppl("distrcompositor", learner = lrn("surv.rpart"), graph_learner = TRUE)
  expect_class(pipe, "GraphLearner")

  p = pipe$train(task)$predict(task)
  expect_prediction_surv(p)
  expect_true("distr" %in% p$predict_types)
})

test_that("survaverager", {
  pipe = mlr3pipelines::ppl("survaverager", learners = list(lrn("surv.kaplan"),
    lrn("surv.kaplan", id = "k2")))
  expect_class(pipe, "Graph")

  pipe = mlr3pipelines::ppl("survaverager", learners = list(lrn("surv.kaplan"),
    lrn("surv.kaplan", id = "k2")),
  graph_learner = TRUE)
  expect_class(pipe, "GraphLearner")

  expect_prediction_surv(pipe$train(task)$predict(task))
})

test_that("survbagging", {
  pipe = mlr3pipelines::ppl("survbagging", lrn("surv.kaplan"), iterations = 2)
  expect_class(pipe, "Graph")
  pipe = mlr3pipelines::ppl("survbagging", lrn("surv.kaplan"), iterations = 2, graph_learner = TRUE)
  expect_class(pipe, "GraphLearner")
  pipe$train(task)
  p = pipe$predict(task)
  expect_prediction_surv(p)
})

test_that("resample survtoregr", {
  pipe = mlr3pipelines::ppl("survtoregr", method = 1, distrcompose = FALSE, graph_learner = TRUE)
  rr = resample(task, pipe, rsmp("cv", folds = 2))
  expect_numeric(rr$aggregate())
})

test_that("survtoregr 1", {
  pipe = mlr3pipelines::ppl("survtoregr", method = 1, distrcompose = FALSE)
  expect_class(pipe, "Graph")
  pipe = mlr3pipelines::ppl("survtoregr", method = 1, distrcompose = FALSE, graph_learner = TRUE)
  expect_class(pipe, "GraphLearner")
  pipe$train(task)
  p = pipe$predict(task)
  expect_prediction_surv(p)

  pipe = mlr3pipelines::ppl("survtoregr", method = 1, distrcompose = TRUE, graph_learner = TRUE)
  expect_class(pipe, "GraphLearner")
  pipe$train(task)
  p = pipe$predict(task)
  expect_prediction_surv(p)
  expect_true("distr" %in% p$predict_types)
})

test_that("survtoregr 2", {
  pipe = mlr3pipelines::ppl("survtoregr", method = 2)
  expect_class(pipe, "Graph")
  pipe = mlr3pipelines::ppl("survtoregr", method = 2, graph_learner = TRUE)
  expect_class(pipe, "GraphLearner")
  pipe$train(task)
  p = pipe$predict(task)
  expect_prediction_surv(p)
  expect_true("distr" %in% p$predict_types)

  pipe = mlr3pipelines::ppl("survtoregr", method = 2, regr_se_learner = lrn("regr.featureless"),
    graph_learner = TRUE)
  expect_class(pipe, "GraphLearner")
  pipe$train(task)
  p = pipe$predict(task)
  expect_prediction_surv(p)
  expect_true("distr" %in% p$predict_types)
})

test_that("survtoregr 3", {
  pipe = mlr3pipelines::ppl("survtoregr", method = 3, distrcompose = FALSE)
  expect_class(pipe, "Graph")
  pipe = mlr3pipelines::ppl("survtoregr", method = 3, distrcompose = FALSE,
                            graph_learner = TRUE)
  expect_class(pipe, "GraphLearner")
  suppressWarnings(pipe$train(task)) # suppress loglik warning
  p = pipe$predict(task)
  expect_prediction_surv(p)

  pipe = mlr3pipelines::ppl("survtoregr", method = 3, distrcompose = TRUE,
                            graph_learner = TRUE)
  expect_class(pipe, "GraphLearner")
  suppressWarnings(pipe$train(task)) # suppress loglik warning
  p = pipe$predict(task)
  expect_prediction_surv(p)
  expect_true("distr" %in% p$predict_types)
})
