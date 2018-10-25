FROM myos6


USER root

# Setup OpenJDK
RUN yum install -y which wget java-1.8.0-openjdk-devel 

# Setup Scala
ENV SCALA_HOME=/root/scala
ENV SCALA_VERSION=2.11.12
ENV PATH=$PATH:$SCALA_HOME/bin

RUN wget https://downloads.typesafe.com/scala/$SCALA_VERSION/scala-$SCALA_VERSION.tgz && \
	tar -xzvf scala-$SCALA_VERSION.tgz -C /root/ && \ 
	mv /root/scala-$SCALA_VERSION $SCALA_HOME && \ 
	rm -rf scala-$SCALA_VERSION.tgz && \ 
	rm -rf $SCALA_HOME/doc

# Setup hadoop
ENV HADOOP_HOME=/root/hadoop
ENV HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HADOOP_HOME/lib/native/:/root/protobuf/lib
ENV PATH=$PATH:/root/protobuf/bin:$HADOOP_HOME/bin:$HADOOP_HOME/sbin:.

RUN wget http://mirrors.ircam.fr/pub/apache/hadoop/common/hadoop-2.9.1/hadoop-2.9.1.tar.gz && \
	tar -xzvf hadoop-2.9.1.tar.gz -C /root/ && \
    mv /root/hadoop-2.9.1 $HADOOP_HOME && \
    rm -rf hadoop-2.9.1.tar.gz && \
    rm -rf $HADOOP_HOME/bin/*.cmd && \
    rm -rf $HADOOP_HOME/sbin/*.cmd && \
    rm -rf $HADOOP_HOME/sbin/*all* && \
    rm -rf $HADOOP_CONF_DIR/*.cmd && \
    rm -rf $HADOOP_CONF_DIR/*.template && \
    rm -rf $HADOOP_CONF_DIR/*.example 

ADD config/hadoop/* $HADOOP_HOME/etc/hadoop/

# Setup Spark
ENV SPARK_HOME=/root/spark
ENV PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin

RUN wget http://mirror.ibcp.fr/pub/apache/spark/spark-2.3.2/spark-2.3.2-bin-hadoop2.7.tgz && \
    tar -xzvf spark-2.3.2-bin-hadoop2.7.tgz -C /root/ && \
    mv /root/spark-2.3.2-bin-hadoop2.7 $SPARK_HOME && \
    rm -rf spark-2.3.2-bin-hadoop2.7.tgz && \
    rm -rf $SPARK_HOME/bin/*.cmd && \
    rm -rf $SPARK_HOME/sbin/*.cmd && \
    rm -rf $SPARK_HOME/examples 

COPY config/spark/* $SPARK_HOME/conf/

# copy env variables
ADD config/other/bashrc /root/.bashrc

CMD [ "sh", "-c", "service sshd start; bash"]