# Retomada do TFG — revisão 01

Data: 16/09/2026. **Rascunho parcial para revisão do autor e dos orientadores; não é uma versão pronta para entrega.**

## O que mudou nesta rodada

O projeto original do Overleaf foi importado sem alterações em um primeiro registro no histórico (`60c7ef0`). A revisão do manuscrito ficou restrita a três arquivos:

- `elemtextuais/1-Introducao.tex`: substituição das instruções do modelo por uma introdução, questão de pesquisa, objetivo geral, cinco objetivos específicos, delimitação e contribuição esperada.
- `configuracao.tex`: título no domínio de desastres naturais, ano 2026 e cinco palavras-chave. Mês de entrega e data de defesa ficaram explicitamente a confirmar; nomes fictícios da banca foram removidos, sem atribuir membros ainda não confirmados.
- `referencias.bib`: correção do ano de Vaswani et al. para 2017, inclusão do DOI do artigo Chg2Cap e inclusão de Gupta et al. (2019) para contextualizar dados de desastres.

Este arquivo de acompanhamento é novo e não participa da compilação do manuscrito. Código do modelo, dados, pesos, figuras, classe do documento e demais seções não foram modificados nesta rodada. O PDF e o ZIP originais em Documentos permanecem preservados.

## Decisões recuperadas do histórico

- O trabalho continua sendo geração de legendas de mudança a partir de imagens bitemporais. A transição de cafezais para desastres naturais muda o domínio, sem exigir o abandono dos fundamentos já escritos.
- O Chg2Cap, de Chang e Ghamisi (2023), permanece como arquitetura de referência. A sequência de trabalho é reproduzir/documentar o baseline urbano e depois realizar adaptações incrementais.
- A contribuição proposta é experimental e aplicada, não a criação de uma arquitetura inédita. Essa formulação precisa ser validada com os orientadores.
- VLMs foram discutidos como alternativas, mas não fazem parte obrigatória da metodologia. Não se planejou nesta revisão trocar a arquitetura ou contratar serviços.
- O referencial deverá seguir a progressão image captioning → change captioning → remote sensing change captioning, reaproveitando o material existente. Sua reorganização ficou para a próxima rodada.
- O relatório recuperado registra início em 2025.2, entrega da parcial, ausência de defesa em 2026.1 e retomada em 2026.2. A anuência da coordenação para aproveitar a parcial é um relato do autor, não um documento institucional verificado.

## Diferenças entre o relato e os arquivos disponíveis

**Título:** foi usado “Geração de Legendas de Mudança em Cenários de Desastres Naturais a partir de Imagens Bitemporais de Sensoriamento Remoto com Transformer”, informado diretamente pelo autor como registrado. O relatório recuperado apresenta uma variante sem o trecho final. Falta conferir o formulário atualizado; a divergência não foi resolvida por suposição.

**Texto existente:** embora o relatório mencione uma introdução desenvolvida, o PDF e o ZIP fornecidos contêm instruções do modelo na seção de introdução. O referencial teórico contém texto substantivo e foi preservado. Se houver outra versão da introdução no Overleaf ou em arquivos anteriores, ela poderá ser comparada com este rascunho.

**Experimentos:** o relato confirma uma execução funcional com dados urbanos. A inspeção local também identificou LEVIR-CC preparado e oito checkpoints distintos. O checkpoint com sufixo `6137` pôde ser carregado e apresentou compatibilidade estrutural com o modelo. Isso não reconstrói as condições de treinamento nem comprova reprodução completa dos resultados do artigo. Os oito checkpoints não foram todos executados.

**Métricas:** números nos nomes dos checkpoints não foram tratados como resultados de teste reproduzidos. O METEOR está desativado no código examinado, e seu zero substituto não é um resultado experimental. O recurso de paráfrases exigido pelo avaliador não foi encontrado; a representação textual enviada às métricas também precisa de revisão antes de qualquer comparação.

**Dados adicionais:** foram localizados arquivos compactados de LEVIR-CC e ONERA em Downloads. Sua presença não comprova a escolha de um conjunto de desastres nem a disponibilidade de legendas adequadas a esse domínio.

## O que ainda está em aberto

1. **Dataset e legendas:** escolher o recorte de desastres, verificar condições de uso, pareamento temporal e anotações; definir se as legendas já existem ou precisarão ser produzidas. xBD aparece na introdução somente como exemplo de dados pré/pós-evento com anotações de danos.
2. **Protocolo de avaliação:** estabelecer divisões de dados, critério de seleção de checkpoint, configurações, métricas e exemplos qualitativos. Evitar compartilhar eventos ou cenas correlacionadas entre treino e teste quando isso puder favorecer a avaliação. Se forem produzidas legendas, documentar autoria, revisão e independência das referências de teste.
3. **Escopo da adaptação:** separar reutilização, correções de execução, preparação dos dados, retreinamento e eventuais mudanças no método. Alterar uma componente por vez, mantendo uma referência comparável.
4. **Escrita restante:** revisar o referencial e substituir os textos do modelo em resumo, desenvolvimento, resultados e conclusões; revisar agradecimentos. Algumas frases do referencial ainda falam em “modelo proposto” e devem ser ajustadas para distinguir a arquitetura dos autores da contribuição deste TFG.
5. **Dados acadêmicos:** conferir formulário/título atualizado, prazo final, mês de entrega, data de defesa e banca. Atualizar o cronograma sem inventar datas.
6. **Recursos e rastreabilidade:** documentar ambiente efetivamente utilizado, origem dos checkpoints e configurações das próximas execuções. A existência de uma GPU no computador não prova seu uso nos experimentos anteriores.

## Próximas rodadas sugeridas

1. O autor revisa a questão de pesquisa e os objetivos desta introdução e leva a delimitação aos orientadores. Em paralelo, pode ser reorganizado o referencial, sem depender de resultados finais.
2. Em uma rodada separada de código, produzir um teste pequeno e rastreável do baseline LEVIR-CC: par de imagens, referência, legenda gerada, checkpoint e configuração. Corrigir problemas de execução/avaliação antes de interpretar métricas.
3. Comparar candidatos de dados de desastres e a viabilidade das legendas. Somente depois consolidar a metodologia e iniciar as adaptações de domínio.

As métricas previstas no histórico são BLEU-1 a BLEU-4, METEOR, ROUGE-L e CIDEr-D. Sua inclusão nos resultados depende de execução válida e registro do protocolo. A análise qualitativa prevista abrange objetos incorretos, mudanças inexistentes, inversões temporais e omissões; não depende de afirmar que já existem resultados finais.

## Fontes da revisão

- Contexto inicial e relatório de continuidade fornecidos pelo autor nesta conversa.
- PDF `TFG_GABRIEL_PEREZ.pdf` e fontes `TFG_GABRIEL_PEREZ.zip`, em `/home/gabriel/Documentos`.
- SHA-256 do ZIP original: `b4d63b29d9bed8f91a47c78f40779ba375455291c6618c7c2b8774644e0b4b61`.
- [Vinyals et al. — Show and Tell](https://arxiv.org/abs/1411.4555): geração de descrições para uma imagem.
- [Park et al. — Robust Change Captioning](https://arxiv.org/abs/1901.02527): descrição de mudanças e distinção de variações irrelevantes.
- [Liu et al. — artigo e implementação dos autores](https://github.com/Chen-Yang-Liu/RSICC): tarefa RSICC e conjunto LEVIR-CC.
- [Chang e Ghamisi — Chg2Cap](https://arxiv.org/abs/2304.01091): arquitetura de referência; [DOI da publicação](https://doi.org/10.1109/TIP.2023.3328224).
- [Vaswani et al. — publicação de 2017](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html): Transformer e confirmação do ano.
- [Gupta et al. — xBD, versão de novembro de 2019](https://arxiv.org/abs/1911.09296): imagens pré/pós-evento e anotações de dano, sem pressupor legendas prontas.

As fontes bibliográficas acima foram conferidas em páginas dos trabalhos, dos autores ou dos anais. Esta conferência é localizada nas afirmações usadas na introdução; não constitui uma revisão sistemática da literatura nem uma auditoria completa da bibliografia original.

## Validação e uso no Overleaf

Foram conferidos a integridade da importação, o limite das alterações aos três arquivos do manuscrito, as chaves de citação, as referências cruzadas, a estrutura básica dos arquivos modificados e os caminhos de arquivos incluídos.

**Não houve compilação local nem validação visual de um novo PDF:** não há compilador LaTeX disponível neste ambiente. As verificações estáticas não garantem a compilação ou a diagramação final. O PDF original fornecido pelo autor não representa esta revisão.

O pacote `output/tcc/TFG_GABRIEL_PEREZ_revisao_01.zip`, relativo à raiz do repositório, contém o projeto completo revisado. Para conferir com segurança, importe-o como um novo projeto no Overleaf e use `main.tex` como documento principal. Compile, confira os avisos e revise especialmente o título longo, a introdução e as referências. Não utilize o PDF como versão final enquanto as demais seções ainda contiverem exemplos do modelo.
