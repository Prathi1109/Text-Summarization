# Text-Summarization
**Summarizing the text from research papers using RNN Encoder Decoder**


![Domain](Domain.png)




In this project, we have discussed an approach of text summarization of research papers using RNN encoder and decoder with the attention mechanism. Since we were initially unable to train on the research papers dataset, which consisted of larger input sequences, we have tried to summarize the NEWS articles (which had com- paratively smaller input sequences). 


The NEWS Model has produced better results qualitatively, when compared to CS Model. The number of instances considered for quantitative analysis is different for NEWS Model (200 instances) and the CS Model (100 instances). 
On the contrary, the results of the quantitative analysis of the CS Model is better than the NEWS Model. The baseline NEWS Model discussed by Lopyrev was trained on 5 million instances (vocabulary=40000 & comprising 236 million words), for 5 days on a GTX 980 Ti GPU . 

It has been observed that the vocabulary for this baseline model has been built over a preexisting corpus. Our NEWS Model was trained on 800 instances (vocabulary=11,276), for 12 hours on Google Colaboratory. Additionally, the vocabulary was built only from the words in training instances, without considering any preexisting corpus. Hence, using a dedi- cated hardware could help in training the models faster for more number of epochs and with more number of training instances. Transitively, this could result in the improvement of the quality of summaries generated. The research papers dataset that has been used was open-source, quite unstructured and unorganized. 


Despite refining the data with multiple filters based on language, topics discussed in the paper, it did not yield data with consistent structure across all the training instances. The selection of a widely used, structured dataset for text summarization would have been wiser. The summarization model could be extended in the future to make it work for longer input sequences with an improved model architecture. There could also be a scope for improving the coherence and readability of the generated summaries by adding sentence generation and/or sentence fusion techniques.



We have utilized the mix-domain approach of transfer learning to implement the domain adaptability of the text summarization model. We have extended the imple- mentation of this approach which was initially discussed by Hua & Wang . The DA Model is a CS Model re-trained with few Engineering documents and uses CS vocabulary. Whereas, the DA-DI Model is a CS Model re-trained with few Engi- neering documents and uses ENG vocabulary. The DA and DA-DI models are an extended implementation of mix-domain approach. It has been observed, both in qualitative and quantitative analysis, that the DA-DI Model has outperformed the DA Model. In addition to this, the DA-DI Model has yielded results at par with the ENG Model, deducing that the DA-DI Model could be used as a possible replacement of the standalone ENG Model. From this, we can determine that the domain adapt- ability of the summarization model with the domain information can be considered over a stand-alone model.
