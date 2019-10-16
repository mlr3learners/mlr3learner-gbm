#' @title Regression gbm Learner
#'
#' @aliases mlr_learners_regr.gbm
#' @format [R6::R6Class] inheriting from [mlr3::LearnerRegr].
#'
#' @description
#' A [mlr3::LearnerRegr] for a classification gbm implemented in [gbm::gbm()] in package \CRANpkg{gbm}.
#'
#' @export
LearnerRegrGBM = R6Class("LearnerRegrGBM", inherit = LearnerRegr,
  public = list(
    initialize = function() {
      ps = ParamSet$new(
        params = list(
          ParamFct$new(id = "distribution", default = "gaussian", levels = c("gaussian", "laplace", "poisson", "tdist", "quantile"), tags = "train"),
          ParamInt$new(id = "n.trees", default = 100L, lower = 1L, tags = c("train", "predict", "importance")),
          ParamInt$new(id = "interaction.depth", default = 1L, lower = 1L, tags = "train"),
          ParamInt$new(id = "n.minobsinnode", default = 10L, lower = 1L, tags = "train"),
          ParamDbl$new(id = "shrinkage", default = 0.001, lower = 0, tags = "train"),
          ParamDbl$new(id = "bag.fraction", default = 0.5, lower = 0, upper = 1, tags = "train"),
          ParamDbl$new(id = "train.fraction", default = 1, lower = 0, upper = 1, tags = "train"),
          ParamInt$new(id = "cv.folds", default = 0L, tags = "train"),
          ParamDbl$new(id = "alpha", default = 0.5, lower = 0, upper = 1, tags = "train")
        )
      )
      ps$add_dep("alpha", "distribution", CondEqual$new("quantile"))

      super$initialize(
        id = "regr.gbm",
        packages = "gbm",
        feature_types = c("integer", "numeric", "factor", "ordered"),
        predict_types = "response",
        param_set = ps,
        properties = c("weights", "importance", "missings")
      )
    },

    train_internal = function(task) {

      # Set to default for predict
      if (is.null(self$param_set$values$n.tress)) {
        self$param_set$values$n.trees = 100
      }

      pars = self$param_set$get_values(tags = "train")
      f = task$formula()
      data = task$data()

      if (!is.null(pars$distribution)) {
        if (pars$distribution == "quantile") {
          alpha = ifelse(is.null(pars$alpha), 0.5, pars$alpha)
          pars$distribution = list(name = "quantile", alpha = alpha)
        }
      }

      if ("weights" %in% task$properties) {
        pars = insert_named(pars, list(weights = task$weights$weight))
      }

      invoke(gbm::gbm, formula = f, data = data, .args = pars)
    },

    predict_internal = function(task) {
      pars = self$param_set$get_values(tags = "predict")
      newdata = task$data(cols = task$feature_names)

      p = invoke(predict, self$model, newdata = newdata, .args = pars)
      PredictionRegr$new(task = task, response = p)
    },

    importance = function() {
      if (is.null(self$model)) {
        stop("No model stored")
      }
      pars = self$param_set$get_values(tags = "importance")

      imp = invoke(gbm::relative.influence, self$model, .args = pars)
      sort(imp, decreasing = TRUE)
    }
  )
)
