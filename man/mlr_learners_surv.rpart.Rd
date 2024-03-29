% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/LearnerSurvRpart.R
\name{mlr_learners_surv.rpart}
\alias{mlr_learners_surv.rpart}
\alias{LearnerSurvRpart}
\title{Rpart Survival Trees Survival Learner}
\description{
Calls \code{\link[rpart:rpart]{rpart::rpart()}}.
\itemize{
\item crank is predicted using \code{\link[rpart:predict.rpart]{rpart::predict.rpart()}}
}

Parameter \code{xval} is set to 0 in order to save some computation time.
Parameter \code{model} has been renamed to \code{keep_model}.
}
\section{Dictionary}{

This \link[mlr3:Learner]{Learner} can be instantiated via the \link[mlr3misc:Dictionary]{dictionary}
\link[mlr3:mlr_learners]{mlr_learners} or with the associated sugar function \link[mlr3:mlr_sugar]{lrn()}:\preformatted{LearnerSurvRpart$new()
mlr_learners$get("surv.rpart")
lrn("surv.rpart")
}
}

\section{Meta Information}{

\itemize{
\item Type: "surv"
\item Predict Types: \verb{crank, distr}
\item Feature Types: \verb{logical, integer, numeric, character, factor, ordered}
\item Properties: \verb{importance, missings, selected_features, weights}
\item Packages: \CRANpkg{mlr3} \CRANpkg{mlr3proba} \CRANpkg{rpart} \CRANpkg{distr6} \CRANpkg{survival}
}
}

\references{
Breiman L, Friedman JH, Olshen RA, Stone CJ (1984).
\emph{Classification And Regression Trees}.
Routledge.
\doi{10.1201/9781315139470}.
}
\seealso{
Other survival learners: 
\code{\link{mlr_learners_surv.coxph}},
\code{\link{mlr_learners_surv.kaplan}}
}
\concept{survival learners}
\section{Super classes}{
\code{\link[mlr3:Learner]{mlr3::Learner}} -> \code{\link[mlr3proba:LearnerSurv]{mlr3proba::LearnerSurv}} -> \code{LearnerSurvRpart}
}
\section{Methods}{
\subsection{Public methods}{
\itemize{
\item \href{#method-new}{\code{LearnerSurvRpart$new()}}
\item \href{#method-importance}{\code{LearnerSurvRpart$importance()}}
\item \href{#method-selected_features}{\code{LearnerSurvRpart$selected_features()}}
\item \href{#method-clone}{\code{LearnerSurvRpart$clone()}}
}
}
\if{html}{
\out{<details ><summary>Inherited methods</summary>}
\itemize{
\item \out{<span class="pkg-link" data-pkg="mlr3" data-topic="Learner" data-id="base_learner">}\href{../../mlr3/html/Learner.html#method-base_learner}{\code{mlr3::Learner$base_learner()}}\out{</span>}
\item \out{<span class="pkg-link" data-pkg="mlr3" data-topic="Learner" data-id="format">}\href{../../mlr3/html/Learner.html#method-format}{\code{mlr3::Learner$format()}}\out{</span>}
\item \out{<span class="pkg-link" data-pkg="mlr3" data-topic="Learner" data-id="help">}\href{../../mlr3/html/Learner.html#method-help}{\code{mlr3::Learner$help()}}\out{</span>}
\item \out{<span class="pkg-link" data-pkg="mlr3" data-topic="Learner" data-id="predict">}\href{../../mlr3/html/Learner.html#method-predict}{\code{mlr3::Learner$predict()}}\out{</span>}
\item \out{<span class="pkg-link" data-pkg="mlr3" data-topic="Learner" data-id="predict_newdata">}\href{../../mlr3/html/Learner.html#method-predict_newdata}{\code{mlr3::Learner$predict_newdata()}}\out{</span>}
\item \out{<span class="pkg-link" data-pkg="mlr3" data-topic="Learner" data-id="print">}\href{../../mlr3/html/Learner.html#method-print}{\code{mlr3::Learner$print()}}\out{</span>}
\item \out{<span class="pkg-link" data-pkg="mlr3" data-topic="Learner" data-id="reset">}\href{../../mlr3/html/Learner.html#method-reset}{\code{mlr3::Learner$reset()}}\out{</span>}
\item \out{<span class="pkg-link" data-pkg="mlr3" data-topic="Learner" data-id="train">}\href{../../mlr3/html/Learner.html#method-train}{\code{mlr3::Learner$train()}}\out{</span>}
}
\out{</details>}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-new"></a>}}
\if{latex}{\out{\hypertarget{method-new}{}}}
\subsection{Method \code{new()}}{
Creates a new instance of this \link[R6:R6Class]{R6} class.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{LearnerSurvRpart$new()}\if{html}{\out{</div>}}
}

}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-importance"></a>}}
\if{latex}{\out{\hypertarget{method-importance}{}}}
\subsection{Method \code{importance()}}{
The importance scores are extracted from the model slot \code{variable.importance}.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{LearnerSurvRpart$importance()}\if{html}{\out{</div>}}
}

\subsection{Returns}{
Named \code{numeric()}.
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-selected_features"></a>}}
\if{latex}{\out{\hypertarget{method-selected_features}{}}}
\subsection{Method \code{selected_features()}}{
Selected features are extracted from the model slot \code{frame$var}.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{LearnerSurvRpart$selected_features()}\if{html}{\out{</div>}}
}

\subsection{Returns}{
\code{character()}.
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-clone"></a>}}
\if{latex}{\out{\hypertarget{method-clone}{}}}
\subsection{Method \code{clone()}}{
The objects of this class are cloneable with this method.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{LearnerSurvRpart$clone(deep = FALSE)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{deep}}{Whether to make a deep clone.}
}
\if{html}{\out{</div>}}
}
}
}
