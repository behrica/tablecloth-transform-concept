(ns try-transform.pca
(:require [tablecloth.api :as tc]
          [tech.v3.ml :as ml]
          [tech.v3.dataset.column-filters :as cf]
          [tech.v3.dataset.modelling :as ds-mod]
          [tech.v3.dataset :as ds]
          [tech.v3.protocols.dataset :as proto-ds]
          [tech.v3.tensor :as t]
          [tech.v3.dataset.tensor :as ds-tensor]
          [tech.v3.libs.xgboost]
          [tech.v3.libs.smile.classification]

          [try-transform.transformers :as tf]
          )
)




(defn ds->x-y [ctx]
  (user/def-let [y (tf/select-columns ctx :target)
                 X (tf/drop-columns ctx :target)
                 ]
    (assoc ctx :y (:dataset y) :dataset (:dataset  X))))

(defn do-pca [ctx]
  (case (:mode ctx)
    :fit (-> ctx
             (tf/pca {:n-components 2} :fit)
             (tf/pca {} :transform)
             )
    :transform (tf/pca ctx {} :transform)))


(comment

  (defn pipeline [ctx]
    (-> ctx
        (tf/rename-columns {:column-64 :target})
        ds->x-y
        do-pca
        ((fn [ctx]
           (tf/append ctx (:y ctx))))
        (tf/set-inference-target :target)
        (tf/categorical->number cf/categorical)
        (tf/train-or-predict :logit {:model-type :smile.classification/logistic-regression})
        ))


  (def ds (tc/dataset "digits.csv.gz" {:header-row? false :key-fn keyword}))

  (def splits (tc/split ds :holdout))

  (def fitted
    (pipeline
     {:mode :fit
      :dataset (:train (first splits))
      }))

  (def prediction
    (pipeline
     {:mode :transform
      :dataset (:test (first splits))
      :pca (:pca fitted)
      :logit (:logit fitted)
      })
    )

  (def label-map (ds-mod/inference-target-label-inverse-map (:dataset prediction)))

  (->>
   (:logit prediction)
   :target
   (map
    #(get label-map  (int %))
    )

   )
  ;; => ("1"
  ;;     "1"
  ;;     "4"
  ;;     "7"
  ;;     "8"
  ;;     "7"
  ;;     .....
  ;;     "5")
  )
;; => nil
