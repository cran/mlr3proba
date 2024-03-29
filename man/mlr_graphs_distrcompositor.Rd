% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/pipelines.R
\name{mlr_graphs_distrcompositor}
\alias{mlr_graphs_distrcompositor}
\alias{pipeline_distrcompositor}
\alias{distrcompositor}
\title{Estimate Survival distr Predict Type Pipeline}
\usage{
pipeline_distrcompositor(
  learner,
  estimator = c("kaplan", "nelson"),
  form = c("aft", "ph", "po"),
  overwrite = FALSE,
  graph_learner = FALSE
)

distrcompositor(...)
}
\arguments{
\item{learner}{\verb{[mlr3::Learner]|[mlr3pipelines::PipeOp]|[mlr3pipelines::Graph]} \cr
Either a \code{Learner} which will be wrapped in \link[mlr3pipelines:mlr_pipeops_learner]{mlr3pipelines::PipeOpLearner}, a \code{PipeOp} which will
be wrapped in \link[mlr3pipelines:Graph]{mlr3pipelines::Graph} or a \code{Graph} itself. Underlying \code{Learner} should be
\link{LearnerSurv}.}

\item{estimator}{\code{character(1)}\cr
One of \code{kaplan} (default) or \code{nelson}, corresponding to the Kaplan-Meier and
Nelson-Aalen estimators respectively. Used to estimate the baseline survival distribution.}

\item{form}{\code{character(1)}\cr
One of \code{aft} (default), \code{ph}, or \code{po}, corresponding to accelerated failure time,
proportional hazards, and proportional odds respectively. Used to determine the form of the
composed survival distribution.}

\item{overwrite}{\code{logical(1)}\cr
If \code{FALSE} (default) then if the \code{learner} already has a \code{distr}, the compositor does nothing.
If \code{TRUE} then the \code{distr} is overwritten by the compositor if
already present, which may be required for changing the prediction \code{distr} from one model form
to another.}

\item{graph_learner}{\code{logical(1)}\cr
If \code{TRUE} returns wraps the \link[mlr3pipelines:Graph]{Graph} as a
\link[mlr3pipelines:mlr_learners_graph]{GraphLearner} otherwise (default) returns as a \code{Graph}.}

\item{...}{\code{ANY}\cr
For use with \code{distrcompositor}, now deprecated.}
}
\value{
\link[mlr3pipelines:Graph]{mlr3pipelines::Graph} or \link[mlr3pipelines:mlr_learners_graph]{mlr3pipelines::GraphLearner}
}
\description{
Wrapper around \link{PipeOpDistrCompositor} to simplify \link[mlr3pipelines:Graph]{Graph} creation.
}
\examples{
\dontrun{
if (requireNamespace("mlr3pipelines", quietly = TRUE) &&
  requireNamespace("rpart", quietly = TRUE)) {
  library("mlr3")
  library("mlr3pipelines")

  task = tsk("rats")
  pipe = ppl(
    "distrcompositor",
    learner = lrn("surv.rpart"),
    estimator = "kaplan",
    form = "ph"
  )
  pipe$train(task)
  pipe$predict(task)
}
}
}
\seealso{
Other pipelines: 
\code{\link{mlr_graphs_crankcompositor}},
\code{\link{mlr_graphs_probregrcompositor}},
\code{\link{mlr_graphs_survaverager}},
\code{\link{mlr_graphs_survbagging}},
\code{\link{mlr_graphs_survtoregr}}
}
\concept{pipelines}
