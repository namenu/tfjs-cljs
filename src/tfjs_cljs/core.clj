(ns tfjs-cljs.core)

(defmacro defvar
  [name value]
  `(def ~name (.variable js/tf ~value)))

(defmacro with-tidy
  "Disposes all Tensors implicitly which are generated while executing body. Sugar of Tf.tidy()."
  [& body]
  `(.tidy js/tf (fn [] ~@body)))

(defmacro with-fit-history
  [args name & body]
  `(.then (tfjs-cljs.layers.models/fit ~@args)
     #(let [~name (cljs.core/js->clj (.-history %) :keywordize-keys true)]
        ~@body)))
