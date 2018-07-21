(defproject examples "0.1.0-SNAPSHOT"
  :min-lein-version "2.7.1"

  :dependencies [[org.clojure/clojure "1.9.0"]
                 [org.clojure/clojurescript "1.10.339"]
                 [org.clojure/core.async  "0.4.474"]
                 [reagent "0.8.1"]
                 [cljsjs/p5 "0.5.8-4"]
                 [namenu/tfjs-cljs "0.1.1-SNAPSHOT"]]

  :plugins [[lein-figwheel "0.5.16"]
            [lein-cljsbuild "1.1.7" :exclusions [[org.clojure/clojure]]]]

  :source-paths ["src"]

  :cljsbuild {:builds
              [{:id "dev"
                :source-paths ["src"]

                :figwheel {:on-jsload "examples.polynomial-regression-core/on-js-reload"
                           :open-urls ["http://localhost:3449/index.html"]}
                :compiler {:main examples.polynomial-regression-core
                           :asset-path "js/compiled/out"
                           :output-to "resources/public/js/compiled/examples.js"
                           :output-dir "resources/public/js/compiled/out"
                           :source-map-timestamp true
                           :preloads [devtools.preload]}}

               {:id "color-classifier"
                :source-paths ["src"]

                :figwheel {:open-urls ["http://localhost:3449/color-classifier.html"]}
                :compiler {:main examples.color-classifier
                           :asset-path "js/compiled/out"
                           :output-to "resources/public/color-classifier/js/compiled/examples.js"
                           :output-dir "resources/public/color-classifier/js/compiled/out"
                           :source-map-timestamp true
                           :preloads [devtools.preload]}}]}

  :profiles {:dev {:dependencies [[binaryage/devtools "0.9.9"]
                                  [figwheel-sidecar "0.5.16"]
                                  [cider/piggieback "0.3.1"]]
                   :source-paths ["src" "dev"]
                   :repl-options {:nrepl-middleware [cider.piggieback/wrap-cljs-repl]}
                   :clean-targets ^{:protect false} ["resources/public/js/compiled"
                                                     :target-path]}})
