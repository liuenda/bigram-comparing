#setwd("C:/Users/liuenda-toshiba/Dropbox/Research/2016.5.11~Reuter/Comparison")

k=10
cluster<-function(k,lang_name){
  dim=200
  df_jp= read.csv(paste('./output/good_vecs_',lang_name,'.csv',sep=''),fileEncoding='utf-8')
  jp10<-subset(df_jp,select=1)
  vec_jp<-df_jp[,2:(dim+1)]
  index <- kmeans(vec_jp, k)
  jp10[2]<-index$cluster
  colnames(jp10)<- c(lang_name,"k10")
  write.csv(jp10, file = paste("./output/",lang_name,k,'_good.csv',sep=''),row.names=FALSE,fileEncoding = 'utf-8')
  return(jp10)
}

jp10=cluster(k,'jp')
en10=cluster(k,'en')