# IA-classifica-comentario

Um modelo BERT para classificar comentários de notícias/vídeos em três categorias de sentimento:

- positivo<br>
- neutro<br>
- negativo

O token [CLS] é utilizado como representação vetorial do texto, seguindo a abordagem padrão de classificação com BERT.

Além disso, o dataset é filtrado para incluir apenas comentários referentes às três classes de tema:

- onça<br>
- fake news<br>
- caseiro

A classificação final considera somente o sentimento, sem incluir prefixo no rótulo.

- Remoção de valores ausentes (NaN)<br>
- Remoção de comentários duplicados<br>
- Seleção apenas das colunas: comment_text, onça, fake news, caseiro<br>
- Combinação dos rótulos em uma única coluna (label_final) contendo somente: positivo, negativo, neutro.

## Modelo
O modelo utilizado é um BERT pré-treinado em português, como:
<br>
"neuralmind/bert-base-portuguese-cased"

Características:<br>
- Entrada: texto comentado<br>
- Tokenização: WordPiece<br>
- Representação: token [CLS]<br>
- Camada final: dense + softmax com 3 neurônios<br>
- Função de perda: CrossEntropyLoss<br>
- Otimizador: AdamW

- ## Treinamento
Durante o treinamento:<br>
- O texto é tokenizado e transformado em tensores (input_ids, attention_mask)<br>
- O token [CLS] é usado para representar todo o comentário<br>
- O modelo aprende a mapear esta representação para uma das três classes finais

## Como usar
Instale as dependencias:
```
!pip install transformers
```
Substitua a variavel 'text' pela mensagem desejada:
```python
text = "Essa notícia é totalmente falsa."

pred = predict(text)
print(pred)   
```
