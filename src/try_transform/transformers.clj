(ns try-transform.transformers
(:require [tablecloth.api :as tc]
          [tech.v3.ml :as ml]
          [tech.v3.dataset.column-filters :as cf]
          [tech.v3.dataset.modelling :as ds-mod]
          [tech.v3.dataset :as ds]
          [tech.v3.protocols.dataset :as proto-ds]
          [tech.v3.libs.xgboost]
          [tech.v3.dataset.math :as ds-math]
          )
)



;;;  adapt exiting function to be transform compliant
(defn drop-rows [pipeline-context rows-selector]
  (assoc pipeline-context
         :dataset
         (tc/drop-rows  (:dataset pipeline-context) rows-selector))
  )

(defn rename-columns [pipeline-context column-mapping]
  (assoc pipeline-context
         :dataset
         (tc/rename-columns (:dataset pipeline-context) column-mapping))
  )

(defn select-columns [pipeline-context column-selector]
  (assoc pipeline-context
         :dataset
         (tc/select-columns (:dataset pipeline-context) column-selector))
  )

(defn drop-columns [pipeline-context column-selector]
  (assoc pipeline-context
         :dataset
         (tc/drop-columns (:dataset pipeline-context) column-selector))
  )

(defn add-or-replace-column [pipeline-context column-name column]
  (assoc pipeline-context
         :dataset
         (tc/add-or-replace-column (:dataset pipeline-context) column-name column))
  )
(defn bind [pipeline-context & datasets]
  (assoc pipeline-context
         :dataset
         (tc/bind (:dataset pipeline-context) datasets))
  )
(defn append [pipeline-context datasets]
  (assoc pipeline-context
         :dataset
         (tc/append (:dataset pipeline-context) datasets))
  )






(defn categorical->number [pipeline-context filter-fn-or-dataset]
  (assoc pipeline-context
         :dataset
         (ds/categorical->number (:dataset pipeline-context) filter-fn-or-dataset))
  )

(defn set-inference-target [pipeline-context target-name-or-seq]
  (assoc pipeline-context
         :dataset
         (ds-mod/set-inference-target (:dataset pipeline-context) target-name-or-seq)))

(defn train-or-predict [pipeline-context id options ]
  (def pipeline-context pipeline-context)
  (def options options)
  (def id id)
  (let [ds (:dataset pipeline-context)
        result (case (:mode pipeline-context)
                 :fit (ml/train ds options)
                 :transform (ml/predict ds (get pipeline-context id) )
                 )]
    (assoc  pipeline-context id result)))

(defn pca [pipeline-context options mode]

  (user/def-let [ds (:dataset pipeline-context)]
    (case mode
      :fit
      (assoc pipeline-context :pca (ds-math/fit-pca ds options))
      :transform
      (assoc pipeline-context :dataset (ds-math/transform-pca ds (:pca pipeline-context))))
    ))
