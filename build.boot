(set-env!
  :source-paths #{"src"}
  :dependencies '[[cljsjs/tfjs "0.12.0-0"]])

(task-options!
  pom {:project 'namenu/tfjs-cljs
       :version "0.2.1"
       :description "ClojureScript wrapper for TensorFlow.js"
       :url "http://github.com/namenu/tfjs-cljs"
       :license {"Eclipse Public License"
                 "http://www.eclipse.org/legal/epl-v10.html"}})
