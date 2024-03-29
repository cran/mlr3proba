#' @importFrom Rcpp sourceCpp
#' @useDynLib mlr3proba
NULL

#' @title Cpp functions
#' @name cpp
#' @description Exported internal cpp functions for developers
#' @keywords internal
NULL

#' @name .c_weight_survival_score
#' @rdname cpp
#' @export
NULL

#' @name .c_get_unique_times
#' @rdname cpp
#' @export
NULL

# nolint start
#' @import checkmate
#' @import data.table
#' @import distr6
#' @import mlr3
#' @import mlr3misc
#' @import paradox
#' @importFrom R6 R6Class
#' @importFrom utils data head tail
#' @importFrom stats reformulate model.matrix model.frame sd predict complete.cases
#' @importFrom survival Surv
"_PACKAGE"
# nolint end

utils::globalVariables(c("ShortName", "ClassName", "missing", "task"))
register_mlr3 = function() {

  x = utils::getFromNamespace("mlr_reflections", ns = "mlr3")

  if (!("surv" %in% x$task_types$type)) {
    x = utils::getFromNamespace("mlr_reflections", ns = "mlr3")
    x$task_types = setkeyv(rbind(x$task_types, rowwise_table(
      ~type, ~package, ~task, ~learner, ~prediction, ~measure,
      "surv", "mlr3proba", "TaskSurv", "LearnerSurv", "PredictionSurv", "MeasureSurv"
    )), "type")
    x$task_col_roles$surv = x$task_col_roles$regr
    x$task_properties$surv = x$task_properties$regr
    x$learner_properties$surv = x$learner_properties$regr
    x$measure_properties$surv = x$measure_properties$regr
    x$learner_predict_types$surv = list(crank = c("crank", "lp", "distr", "response"),
      distr = c("crank", "lp", "distr", "response"),
      lp = c("crank", "lp", "distr", "response"),
      response = c("crank", "lp", "distr", "response"))
    x$default_measures$surv = "surv.cindex"
  }

  if (!("dens" %in% x$task_types$type)) {
    x = utils::getFromNamespace("mlr_reflections", ns = "mlr3")
    x$task_types = setkeyv(rbind(x$task_types, rowwise_table(
      ~type, ~package, ~task, ~learner, ~prediction, ~measure,
      "dens", "mlr3proba", "TaskDens", "LearnerDens", "PredictionDens", "MeasureDens"
    )), "type")
    x$task_col_roles$dens = c("feature", "target", "label", "order", "group", "weight", "stratum")
    x$task_properties$dens = x$task_properties$regr
    x$learner_properties$dens = x$learner_properties$regr
    x$measure_properties$dens = x$measure_properties$regr
    x$learner_predict_types$dens = list(
      pdf = c("pdf", "cdf", "distr"),
      cdf = c("pdf", "cdf", "distr"),
      distr = c("pdf", "cdf", "distr"))
    x$default_measures$dens = "dens.logloss"
  }

  # tasks
  x = utils::getFromNamespace("mlr_tasks", ns = "mlr3")
  x$add("precip", load_precip)
  x$add("faithful", load_faithful)
  x$add("rats", load_rats)
  x$add("lung", load_lung)
  x$add("actg", load_actg)
  x$add("gbcs", load_gbcs)
  x$add("grace", load_grace)
  x$add("whas", load_whas)
  x$add("unemployment", load_unemployment)

  # generators
  x = utils::getFromNamespace("mlr_task_generators", ns = "mlr3")
  x$add("simdens", TaskGeneratorSimdens)
  x$add("simsurv", TaskGeneratorSimsurv)

  # learners
  x = utils::getFromNamespace("mlr_learners", ns = "mlr3")

  x$add("dens.hist", LearnerDensHistogram)
  x$add("dens.kde", LearnerDensKDE)

  x$add("surv.coxph", LearnerSurvCoxPH)
  x$add("surv.kaplan", LearnerSurvKaplan)
  x$add("surv.rpart", LearnerSurvRpart)

  # measures
  x = utils::getFromNamespace("mlr_measures", ns = "mlr3")

  x$add("dens.logloss", MeasureDensLogloss)

  # x$add("regr.logloss", MeasureRegrLogloss)

  x$add("surv.graf", MeasureSurvGraf)
  x$add("surv.brier", MeasureSurvGraf)
  x$add("surv.schmid", MeasureSurvSchmid)
  x$add("surv.logloss", MeasureSurvLogloss)
  x$add("surv.rcll", MeasureSurvRCLL)
  x$add("surv.intlogloss", MeasureSurvIntLogloss)

  x$add("surv.cindex", MeasureSurvCindex)

  x$add("surv.dcalib", MeasureSurvDCalibration)
  x$add("surv.calib_beta", MeasureSurvCalibrationBeta)
  x$add("surv.calib_alpha", MeasureSurvCalibrationAlpha)

  x$add("surv.nagelk_r2", MeasureSurvNagelkR2)
  x$add("surv.oquigley_r2", MeasureSurvOQuigleyR2)
  x$add("surv.xu_r2", MeasureSurvXuR2)

  x$add("surv.chambless_auc", MeasureSurvChamblessAUC)
  x$add("surv.hung_auc", MeasureSurvHungAUC)
  x$add("surv.uno_auc", MeasureSurvUnoAUC)
  x$add("surv.song_auc", MeasureSurvSongAUC)

  x$add("surv.uno_tpr", MeasureSurvUnoTPR)
  x$add("surv.song_tpr", MeasureSurvSongTPR)

  x$add("surv.uno_tnr", MeasureSurvUnoTNR)
  x$add("surv.song_tnr", MeasureSurvSongTNR)

  x$add("surv.rmse", MeasureSurvRMSE)
  x$add("surv.mse", MeasureSurvMSE)
  x$add("surv.mae", MeasureSurvMAE)
}
register_mlr3pipelines = function() {
  mlr3pipelines::add_class_hierarchy_cache(c("PredictionSurv", "Prediction"))

  x = utils::getFromNamespace("mlr_pipeops", ns = "mlr3pipelines")

  # deprecated
  x$add("distrcompose", PipeOpDistrCompositor)
  x$add("crankcompose", PipeOpCrankCompositor)

  x$add("trafotask_regrsurv", PipeOpTaskRegrSurv)
  x$add("trafotask_survregr", PipeOpTaskSurvRegr)
  x$add("trafopred_regrsurv", PipeOpPredRegrSurv)
  x$add("trafopred_survregr", PipeOpPredSurvRegr)

  x$add("compose_distr", PipeOpDistrCompositor)
  x$add("compose_crank", PipeOpCrankCompositor)
  x$add("compose_probregr", PipeOpProbregrCompositor)

  x$add("survavg", PipeOpSurvAvg)

  x = utils::getFromNamespace("mlr_graphs", ns = "mlr3pipelines")
  x$add("distrcompositor", pipeline_distrcompositor)
  x$add("crankcompositor", pipeline_crankcompositor)
  x$add("probregrcompositor", pipeline_probregrcompositor)
  x$add("survaverager", pipeline_survaverager)
  x$add("survbagging", pipeline_survbagging)
  x$add("survtoregr", pipeline_survtoregr)
}

.onLoad = function(libname, pkgname) { # nolint
  register_mlr3()
  if (requireNamespace("mlr3pipelines", quietly = TRUE)) {
    register_mlr3pipelines()
  }


  setHook(packageEvent("mlr3", "onLoad"), function(...) register_mlr3(), action = "append")
  setHook(packageEvent("mlr3pipelines", "onLoad"), function(...) register_mlr3pipelines(),
    action = "append")
}

.onUnload = function(libpath) { # nolint
  event = packageEvent("mlr3", "onLoad")
  hooks = getHook(event)
  pkgname = vapply(hooks[-1], function(x) environment(x)$pkgname, NA_character_)
  setHook(event, hooks[pkgname != "mlr3proba"], action = "replace")

  event = packageEvent("mlr3pipelines", "onLoad")
  hooks = getHook(event)
  pkgname = vapply(hooks[-1], function(x) environment(x)$pkgname, NA_character_)
  setHook(event, hooks[pkgname != "mlr3proba"], action = "replace")

  library.dynam.unload("mlr3proba", libpath)
}

leanify_package()
