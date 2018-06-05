(ns tfjs-cljs.core)

(defmacro defvar
  [name value]
  `(def ~name (.variable js/tf ~value)))

(defmacro with-tidy
  "Note: Returning tensor might leak your memory."
  [& body]
  `(.tidy js/tf (fn [] ~@body)))
