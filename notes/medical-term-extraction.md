F-measure

Unigrams are better

Unstructured Natural Language Text

Paper's object: medical term extraction

**Conditional Random Fields** (CRF) - разновидность метода Марковских Случайных Полей (Markov Random Field) [[статья на хабре](https://habrahabr.ru/post/241317/)]

C-Value/NC-Value

Рассматривается `medical term extraction` как извлечение ключевых фраз или document summarization task, анализируя граф совместной встречаемости (co-occurence graph) с использованием алгоритма TextRank.

- `Sequence mining` основывается на том, что слова в сложных медицинских текстах часто встречаются в порядке, за который отвечает (?) *lexical level*.
- Для `C-Value` необходима фильтрация *syntactical level* по части речи для определения (?) фраз и их частот. TextRank уделяет много внимания графу, основанному на совместном отношении *structural level*.

Для объединения трёх ranking scores был разработан Genetic Algorithm (GA).

### Related Work
##### Candidate Term Generation
- Linguistic Filters
- N-gram
