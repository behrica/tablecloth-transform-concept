(ns try-transform.main
  (:require [tablecloth.api :as tc]
            [tech.v3.ml :as ml]
            [tech.v3.dataset.column-filters :as cf]
            [tech.v3.dataset.modelling :as ds-mod]
            [tech.v3.dataset :as ds]
            [tech.v3.protocols.dataset :as proto-ds]
            [tech.v3.libs.xgboost]

            [try-transform.transformers :as tf]
            )

  )






(comment


  (def ds (tc/dataset "https://raw.githubusercontent.com/techascent/tech.ml/master/test/data/iris.csv"))



;;;  just fo demo, train=test
  (def train-ds ds)
  (def test-ds ds)


  (defn run-pipeline [ds context]
    (-> (merge context {:dataset ds})
        (tf/drop-rows 2) ;; example for arbitray tabclecoth transformation
        (tf/categorical->number cf/categorical)
        (tf/set-inference-target "species")
        (tf/train-or-predict :xgboost-hinge {:model-type :xgboost/binary-hinge-loss})
        (tf/train-or-predict :xgboost-class {:model-type :xgboost/classification})
        ))


  ;; fit thge pipeline (including train)
  (def fit-result
    (run-pipeline train-ds {:mode :fit}))

;;;  transform (= predict ) on test
  (def transform-result
    (run-pipeline test-ds (merge fit-result {:mode :transform})))


  )
