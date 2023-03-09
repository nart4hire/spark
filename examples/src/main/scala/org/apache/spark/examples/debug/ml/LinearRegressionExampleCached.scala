/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// scalastyle:off println
package org.apache.spark.examples.debug.ml

import scopt.OptionParser

import org.apache.spark.examples.mllib.AbstractParams
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.Transformer
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.storage.StorageLevel

/**
 * An example runner for linear regression with elastic-net (mixing L1/L2) regularization.
 * Run with
 * {{{
 * bin/run-example debug.ml.LinearRegressionExampleCached [options]
 * }}}
 * A synthetic dataset can be found at `data/mllib/sample_linear_regression_data.txt` which can be
 * trained by
 * {{{
 * bin/run-example debug.ml.LinearRegressionExampleCached --regParam 0.15 --elasticNetParam 1.0 \
 *   data/mllib/sample_linear_regression_data.txt
 * }}}
 * If you use it as a template to create your own app, please use `spark-submit` to submit your app.
 */
object LinearRegressionExampleCached {

  case class Params(
      input: String = null,
      testInput: String = "",
      dataFormat: String = "libsvm",
      regParam: Double = 0.0,
      elasticNetParam: Double = 0.0,
      maxIter: Int = 100,
      tol: Double = 1E-6,
      fracTest: Double = 0.2) extends AbstractParams[Params]

  def main(args: Array[String]): Unit = {
    val defaultParams = Params()

    val parser = new OptionParser[Params]("LinearRegressionExampleCached") {
      head("LinearRegressionExampleCached: an example Linear Regression with Elastic-Net app.")
      opt[Double]("regParam")
        .text(s"regularization parameter, default: ${defaultParams.regParam}")
        .action((x, c) => c.copy(regParam = x))
      opt[Double]("elasticNetParam")
        .text(s"ElasticNet mixing parameter. For alpha = 0, the penalty is an L2 penalty. " +
        s"For alpha = 1, it is an L1 penalty. For 0 < alpha < 1, the penalty is a combination of " +
        s"L1 and L2, default: ${defaultParams.elasticNetParam}")
        .action((x, c) => c.copy(elasticNetParam = x))
      opt[Int]("maxIter")
        .text(s"maximum number of iterations, default: ${defaultParams.maxIter}")
        .action((x, c) => c.copy(maxIter = x))
      opt[Double]("tol")
        .text(s"the convergence tolerance of iterations, Smaller value will lead " +
        s"to higher accuracy with the cost of more iterations, default: ${defaultParams.tol}")
        .action((x, c) => c.copy(tol = x))
      opt[Double]("fracTest")
        .text(s"fraction of data to hold out for testing. If given option testInput, " +
        s"this option is ignored. default: ${defaultParams.fracTest}")
        .action((x, c) => c.copy(fracTest = x))
      opt[String]("testInput")
        .text(s"input path to test dataset. If given, option fracTest is ignored." +
        s" default: ${defaultParams.testInput}")
        .action((x, c) => c.copy(testInput = x))
      opt[String]("dataFormat")
        .text("data format: libsvm (default), dense (deprecated in Spark v1.1)")
        .action((x, c) => c.copy(dataFormat = x))
      arg[String]("<input>")
        .text("input path to labeled examples")
        .required()
        .action((x, c) => c.copy(input = x))
      checkConfig { params =>
        if (params.fracTest < 0 || params.fracTest >= 1) {
          failure(s"fracTest ${params.fracTest} value incorrect; should be in [0,1).")
        } else {
          success
        }
      }
    }

    parser.parse(args, defaultParams) match {
      case Some(params) => run(params)
      case _ => sys.exit(1)
    }
  }

  /** Load a dataset from the given path, using the given format */
  private[ml] def loadData(
      spark: SparkSession,
      path: String,
      format: String,
      expectedNumFeatures: Option[Int] = None): DataFrame = {
    import spark.implicits._

    format match {
      case "dense" => MLUtils.loadLabeledPoints(spark.sparkContext, path).toDF()
      case "libsvm" => expectedNumFeatures match {
        case Some(numFeatures) => spark.read.option("numFeatures", numFeatures.toString)
          .format("libsvm").load(path)
        case None => spark.read.format("libsvm").load(path)
      }
      case _ => throw new IllegalArgumentException(s"Bad data format: $format")
    }
  }

  /**
   * Load training and test data from files.
   * @param input  Path to input dataset.
   * @param dataFormat  "libsvm" or "dense"
   * @param testInput  Path to test dataset.
   * @param algo  Classification or Regression
   * @param fracTest  Fraction of input data to hold out for testing. Ignored if testInput given.
   * @return  (training dataset, test dataset)
   */
  private[ml] def loadDatasets(
      input: String,
      dataFormat: String,
      testInput: String,
      algo: String,
      fracTest: Double): (DataFrame, DataFrame) = {
    val spark = SparkSession
      .builder
      .getOrCreate()

    // Load training data
    val origExamples: DataFrame = loadData(spark, input, dataFormat).persist(StorageLevel.MEMORY_ONLY)

    // Load or create test set
    val dataframes: Array[DataFrame] = if (testInput != "") {
      // Load testInput.
      val numFeatures = origExamples.first().getAs[Vector](1).size
      val origTestExamples: DataFrame =
        loadData(spark, testInput, dataFormat, Some(numFeatures))
      Array(origExamples, origTestExamples)
    } else {
      // Split input into training, test.
      origExamples.randomSplit(Array(1.0 - fracTest, fracTest), seed = 12345)
    }

    val training = dataframes(0).persist(StorageLevel.MEMORY_ONLY)
    val test = dataframes(1).persist(StorageLevel.MEMORY_ONLY)

    val numTraining = training.count()
    val numTest = test.count()
    val numFeatures = training.select("features").first().getAs[Vector](0).size
    println("Loaded data:")
    println(s"  numTraining = $numTraining, numTest = $numTest")
    println(s"  numFeatures = $numFeatures")

    (training, test)
  }

  /**
   * Evaluate the given RegressionModel on data. Print the results.
   * @param model  Must fit RegressionModel abstraction
   * @param data  DataFrame with "prediction" and labelColName columns
   * @param labelColName  Name of the labelCol parameter for the model
   *
   * TODO: Change model type to RegressionModel once that API is public. SPARK-5995
   */
  private[ml] def evaluateRegressionModel(
      model: Transformer,
      data: DataFrame,
      labelColName: String): Unit = {
    val fullPredictions = model.transform(data).persist(StorageLevel.MEMORY_ONLY)
    val predictions = fullPredictions.select("prediction").rdd.map(_.getDouble(0)).cache()
    val labels = fullPredictions.select(labelColName).rdd.map(_.getDouble(0)).cache()
    val RMSE = new RegressionMetrics(predictions.zip(labels)).rootMeanSquaredError
    println(s"  Root mean squared error (RMSE): $RMSE")
  }

  def run(params: Params): Unit = {
    val spark = SparkSession
      .builder
      .appName(s"LinearRegressionExampleCached with $params")
      .getOrCreate()

    println(s"LinearRegressionExampleCached with parameters:\n$params")

    // Load training and test data and cache it.
    val (training: DataFrame, test: DataFrame) = loadDatasets(params.input,
      params.dataFormat, params.testInput, "regression", params.fracTest)

    val lir = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setRegParam(params.regParam)
      .setElasticNetParam(params.elasticNetParam)
      .setMaxIter(params.maxIter)
      .setTol(params.tol)

    // Train the model
    val startTime = System.nanoTime()
    val lirModel = lir.fit(training)
    val elapsedTime = (System.nanoTime() - startTime) / 1e9
    println(s"Training time: $elapsedTime seconds")

    // Print the weights and intercept for linear regression.
    println(s"Weights: ${lirModel.coefficients} Intercept: ${lirModel.intercept}")

    println("Training data results:")
    evaluateRegressionModel(lirModel, training, "label")
    println("Test data results:")
    evaluateRegressionModel(lirModel, test, "label")

    spark.stop()
  }
}
// scalastyle:on println
