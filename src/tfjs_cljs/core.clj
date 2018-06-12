(ns tfjs-cljs.core)

(defmacro defvar
  [name value]
  `(def ~name (.variable js/tf ~value)))

(defmacro with-tidy
  "Note: Returning tensor might leak your memory."
  [& body]
  `(.tidy js/tf (fn [] ~@body)))

(defmacro with-fit-history
  [args name & body]
  `(.then (tfjs-cljs.models/fit ~@args)
     #(let [~name (cljs.core/js->clj (.-history %) :keywordize-keys true)]
        ~@body)))
