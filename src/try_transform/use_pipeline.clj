
(ns try-transform.use-pipeline
  (:require [tablecloth.api :as tc]
            [tablecloth.pipeline :as pipe]
            [tech.v3.ml.pipeline :as ml-pipe]
            [tech.v3.dataset.column-filters :as cf]
            [tech.v3.dataset.modelling :as ds-mod]
            [tech.v3.dataset :as ds]
            [tech.v3.protocols.dataset :as proto-ds]
            [tech.v3.libs.smile.classification]))


;; adapt needed dataset functions to pipeline
(defn set-inference-target [column-selector]
  (fn [ctx] (assoc ctx :dataset (ds-mod/set-inference-target (:dataset ctx) column-selector))  ))
(defn categorical->number [filter-fn]
  (fn [ctx] (assoc ctx :dataset (ds/categorical->number (:dataset ctx) filter-fn))  ))









;; read data
(def ds
  (tc/dataset "https://raw.githubusercontent.com/techascent/tech.ml/master/test/data/iris.csv" {:key-fn keyword}))



;; split data
(def split
  (first (tc/split ds :holdout )))
(def train-ds (:train split))


(def test-ds
  (:test split))



;; make pipeline
(defn make-my-pipeline [model-options]
  (pipe/pipeline
   (pipe/shuffle)                       ;; can be any number of tc operations
   (set-inference-target :species)
   (categorical->number cf/categorical)
   (ml-pipe/model (merge model-options
                         {:model-type :smile.classification/logistic-regression}))))


(def model-hyper-params [{:lambda 0.1}
                         {:lamba 0.2}])


;; train and evaluate model for all hyper parameter combination s
(def trained-models
  (map
   (fn [model-options]
     (let [p (make-my-pipeline model-options)
           fitted (p {:dataset train-ds
                      :mode :fit})
           predicted (p (merge fitted
                               {:dataset test-ds
                                :mode :transform}
                               ))
           accuracy 0.2             ;todoi, calculate from predicted and test-ds
           ]
       {:accuracy accuracy
        :fitted-model fitted
        }))

   model-hyper-params))
