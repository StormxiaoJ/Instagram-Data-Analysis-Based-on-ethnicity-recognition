
object ReadTxt{
	def main(args: Array[String]): Unit = {
		val conf = new SparkConf();
		conf.set("spark.master","local");
		conf.set("spark.app.name","Count");
		val sc = new SparkContext(conf);

		// read data
		//remeber to modify the InputFilePath and OutputFilePath
		val textFileRdd = sc.textFile(InputFilePath)
		val fRdd = textFileRdd.flatMap {_.split("\t")}
		val mrdd = fRdd.map{(_,1)}
		val finalRDD = mrdd.reduceByKey(_+_)

		finalRDD.saveAsTextFile(OutputFilePath) 
	}
}