#' @template surv_measure
#' @templateVar title Nagelkerke's R2
#' @templateVar fullname MeasureSurvNagelkR2
#' @description
#' Calls [survAUC::Nagelk()].
#'
#' Assumes Cox PH model specification.
#'
#' @template measure_survAUC
#'
#' @references
#' `r format_bib("nagelkerke_1991")`
#'
#' @family R2 survival measures
#' @family lp survival measures
#' @export
MeasureSurvNagelkR2 = R6Class("MeasureSurvNagelkR2",
  inherit = MeasureSurv,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function() {
      super$initialize(
        id = "surv.nagelk_r2",
        range = 0:1,
        minimize = FALSE,
        packages = "survAUC",
        predict_type = "lp",
        properties = c("requires_task", "requires_train_set"),
        label = "Nagelkerke's R2",
        man = "mlr3proba::mlr_measures_surv.nagelk_r2"
      )
    }
  ),

  private = list(
    .score = function(prediction, task, train_set, ...) {
      surv_train = task$truth(train_set)

      survAUC::Nagelk(surv_train, prediction$lp, rep(0, length(prediction$lp)))
    }
  )
)
