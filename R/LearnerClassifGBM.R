#' @title Classification Gradient Boosting Machine Learner
#'
#' @name mlr_learners_classif.gbm
#'
#' @description
#' Classification gradient boosting machine models.
#' Calls [gbm::gbm()] from package \CRANpkg{gbm}.
#'
#' @templateVar id classif.gbm
#' @template section_dictionary_learner
#'
#' @export
#' @template seealso_learner
#' @template example
LearnerClassifGBM = R6Class("LearnerClassifGBM", inherit = LearnerClassif,
  public = list(

    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function() {
      ps = ParamSet$new(
        params = list(
          ParamFct$new(id = "distribution", default = "bernoulli",
            levels = c("bernoulli", "adaboost", "huberized", "multinomial"),
            tags = "train"),
          ParamInt$new(id = "n.trees", default = 100L, lower = 1L,
            tags = c("train", "predict", "importance")),
          ParamInt$new(id = "interaction.depth", default = 1L, lower = 1L,
            tags = "train"),
          ParamInt$new(id = "n.minobsinnode", default = 10L, lower = 1L,
            tags = "train"),
          ParamDbl$new(id = "shrinkage", default = 0.001, lower = 0,
            tags = "train"),
          ParamDbl$new(id = "bag.fraction", default = 0.5, lower = 0, upper = 1,
            tags = "train"),
          ParamDbl$new(id = "train.fraction", default = 1, lower = 0, upper = 1,
            tags = "train"),
          ParamInt$new(id = "cv.folds", default = 0L, tags = "train"),
          # Set to FALSE to reduce memory requirements
          ParamLgl$new(id = "keep.data", default = FALSE, tags = "train"),
          ParamLgl$new(id = "verbose", default = FALSE, tags = "train"),
          # Set to 1 to suppress parallelization by the package
          ParamInt$new(id = "n.cores", default = 1, tags = "train"),
          ParamUty$new(id = "var.monotone", tags = "train")
        )
      )
      ps$values = list(keep.data = FALSE, n.cores = 1)

      super$initialize(
        id = "classif.gbm",
        packages = "gbm",
        feature_types = c("integer", "numeric", "factor", "ordered"),
        predict_types = c("response", "prob"),
        param_set = ps,
        properties = c("weights", "twoclass", "multiclass", "importance",
          "missings"),
        man = "mlr3learners.gbm::mlr_learners_regr.gbm"
      )
    },

    #' @description
    #' The importance scores are extracted by `gbm::relative.influence()` from
    #' the model.
    #'
    #' @return Named `numeric()`.
    importance = function() {
      if (is.null(self$model)) {
        stop("No model stored")
      }
      pars = self$param_set$get_values(tags = "importance")

      imp = invoke(gbm::relative.influence, self$model, .args = pars)
      sort(imp, decreasing = TRUE)
    }
  ),

  private = list(
    .train = function(task) {

      # Set to default for predict
      if (is.null(self$param_set$values$n.tress)) {
        self$param_set$values$n.trees = 100
      }

      pars = self$param_set$get_values(tags = "train")
      f = task$formula()
      data = task$data()

      if ("weights" %in% task$properties) {
        pars = insert_named(pars, list(weights = task$weights$weight))
      }

      if (task$properties %in% "twoclass") {
        data[[task$target_names]] =
          as.numeric(data[[task$target_names]] == task$positive)
      }

      invoke(gbm::gbm, formula = f, data = data, .args = pars)
    },

    .predict = function(task) {
      pars = self$param_set$get_values(tags = "predict")
      newdata = task$data(cols = task$feature_names)

      p = invoke(predict, self$model, newdata = newdata, type = "response",
        .args = pars)

      if (self$predict_type == "response") {
        if (task$properties %in% "twoclass") {
          p = as.factor(ifelse(p > 0.5, task$positive, task$negative))
          PredictionClassif$new(task = task, response = p)
        } else {
          ind = apply(p, 1, which.max)
          cns = colnames(p)
          PredictionClassif$new(task = task,
            response = factor(cns[ind], levels = cns))
        }
      } else {
        if (task$properties %in% "twoclass") {
          p = matrix(c(p, 1 - p), ncol = 2L, nrow = length(p))
          colnames(p) = task$class_names
          PredictionClassif$new(task = task, prob = p)
        } else {
          PredictionClassif$new(task = task, prob = p[, , 1L])
        }
      }
    }
  )
)
