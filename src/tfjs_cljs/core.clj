(ns tfjs-cljs.core)

(defmacro with-tidy
  [& body]
  `(.tidy js/tf (fn [] ~@body)))
