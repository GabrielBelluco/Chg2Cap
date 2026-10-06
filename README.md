# Chg2Cap e Qwen para legendas de mudanças

Implementação do Chg2Cap de Shizhen Chang e Pedram Ghamisi, com integração
RSCC/EBD e comparação com Qwen2.5-VL-3B-Instruct. A interface para novos
experimentos é `python -m captioning`. Qwen é usado sem fine-tuning.

## Organização

| Pasta | Conteúdo |
|---|---|
| `captioning/` | Interface, preparação, treino Chg2Cap, Qwen e métricas |
| `configs/` | Configurações de exemplo, requisitos e restrições por ambiente |
| `tests/` | Testes de software com dados sintéticos e objetos simulados |
| `data/RSCC/rscc.py` | Leitura de anotações, vocabulário e carregador de pares |
| `model/`, `eval_func/` | Arquitetura e avaliadores originais usados pelo pacote |
| `Figure/`, `LICENSE.txt` | Figuras, atribuição e licença do projeto original |
| `tcc/` | Fontes do trabalho acadêmico, com revisão separada do código |

Dados, ambientes, pesos e saídas em `output/` ficam locais. `train.py`,
`test.py`, `preprocess_data.py` e `requirements.txt` na raiz pertencem ao fluxo
original LEVIR/Dubai; a instalação descrita abaixo usa os arquivos em `configs/`.

Os scripts históricos `rscc.py`, `qwen.py`, `scripts/` e a ferramenta METEOR em
`output/metrics-tools/` permanecem na máquina de pesquisa, fora da seleção
corrente. Seus caminhos e hashes sustentam experimentos concluídos. Não são
necessários para iniciar um novo experimento. Retomar um **checkpoint histórico**
exige também suas fontes originais, manifestos e histórico: o código confere
esses hashes e recusa divergências. Esta seleção não entrega os artefatos
necessários para reproduzir exatamente os resultados antigos.

## Instalação

Ambiente observado: Linux, Python 3.12, AMD RX 6800 de 16 GB e PyTorch
2.4.0+rocm6.1. O driver/ROCm deve estar funcional. Os executores de modelos
exigem GPU; execução completa em outra GPU ou plataforma ainda não foi
verificada. O nome `cuda` no PyTorch também identifica dispositivos ROCm.

Use a raiz do clone como diretório de trabalho. Crie os ambientes abaixo
**somente se os destinos ainda não existirem**. Eles ficam separados porque
Qwen e os avaliadores semânticos usam versões diferentes de Transformers.

```bash
python3.12 -m venv venv_rocm
venv_rocm/bin/python -m pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/rocm6.1
venv_rocm/bin/python -m pip install -c configs/rocm_constraints.txt -r configs/training_requirements.txt
venv_rocm/bin/python -m pip check

python3.12 -m venv .venv/rscc-metrics
.venv/rscc-metrics/bin/python -m pip install -r configs/meteor_requirements.txt
.venv/rscc-metrics/bin/python -m nltk.downloader -d .venv/rscc-metrics/nltk_data wordnet punkt_tab omw-1.4

python3.12 -m venv .venv/rscc-semantic-gpu
.venv/rscc-semantic-gpu/bin/python -m pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/rocm6.1
.venv/rscc-semantic-gpu/bin/python -m pip install -c configs/rocm_constraints.txt -r configs/semantic_requirements.txt
.venv/rscc-semantic-gpu/bin/python -m pip check
```

Esses comandos usam rede. As versões fixam as dependências diretas observadas,
sem constituir um lock completo de dependências transitivas. As constraints
recusam substituir o Torch ROCm silenciosamente. Registre `pip freeze` junto
à execução. Não instale por cima dos ambientes dos experimentos concluídos.

Diagnóstico do ambiente de treino, sem executar um modelo:

```bash
venv_rocm/bin/python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())"
```

## Dados e pesos

EBD contém 18.215 pares na revisão RSCC
[`791a00849c3e684f54df38291cf35a7d828d7004`](https://huggingface.co/datasets/BiliSakura/RSCC/tree/791a00849c3e684f54df38291cf35a7d828d7004).
Obtenha dessa revisão `RSCC_qvq.jsonl` e as cinco partes `EBD.tar.gz-part-0`
a `EBD.tar.gz-part-4` da pasta `EBD/`. Guarde as anotações em
`data/RSCC/raw/RSCC_qvq.jsonl` e as partes em `data/RSCC/full/parts/`.
SHA-256 das anotações:
`96be5cd62f51910caabee011e00ce973fd817dbd256adb338cca154a7924421e`.
O extrator verifica os hashes das cinco partes antes de gravar imagens.

As partes somam cerca de 14,49 GB compactados. Reserve espaço adicional para
imagens extraídas, pesos, ambientes e checkpoints; confira `df -h .`.
RSCC completo inclui xBD e exige suas condições de acesso, imagens e lista
oficial de teste; este roteiro usa EBD. Consulte as
[instruções da fonte](https://huggingface.co/datasets/BiliSakura/RSCC).

Se as imagens ainda não estiverem extraídas, escolha um destino novo:

```bash
venv_rocm/bin/python -B -m captioning extract --parts-dir data/RSCC/full/parts --annotations data/RSCC/raw/RSCC_qvq.jsonl --output data/RSCC/full/images
venv_rocm/bin/python -B -m captioning prepare --annotations data/RSCC/raw/RSCC_qvq.jsonl --image-root data/RSCC/full/images --scope EBD --output data/RSCC/prepared/ebd-novo-v1
```

A preparação valida imagens, agrupa cenas/pixels para impedir sobreposição
entre splits e constrói o vocabulário somente com treino. Sem
`--preserve-manifest`, produz uma nova divisão interna; isso não reproduz o
split dos experimentos anteriores nem o protocolo oficial RSCC. Para manter
atribuições antigas, passe os manifestos existentes após essa opção.
Uma falha pode deixar uma extração parcial: inspecione-a e use outro destino.

Edite `prepared` e `image_root` em `configs/captioning_ebd.json` e
`configs/captioning_qwen.json`. Para Qwen, atualize também `expected_counts`
com as contagens de `audit.json` e `manifest_sha256` com o hash do manifesto:

```bash
sha256sum data/RSCC/prepared/ebd-novo-v1/manifest.jsonl
```

Os exemplos de configuração contêm caminhos e hash do conjunto histórico
local; não presumem que esses arquivos estejam presentes em um clone.
A capacidade das referências considera o conjunto de anotações completo,
seguindo a preparação existente; não representa escolha por desempenho no teste.

Chg2Cap usa ResNet101 ImageNet. Obtenha previamente seu peso no cache Torch:

```bash
venv_rocm/bin/python -c "from torchvision.models import ResNet101_Weights; ResNet101_Weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True)"
```

Os avaliadores usam `sentence-transformers/sentence-t5-xxl` na revisão
`97f38f2f69860f732f602829da6df76ee5c4c0f9` e `FacebookAI/roberta-large` na revisão
`722cf37b1afa9454edce342e7895e588b6ff1d59`. O primeiro ocupa cerca de 9,73 GB.
O comando sem `--execute` apenas apresenta o plano:

```bash
python3 -B -m captioning download-metrics --model st5
.venv/rscc-semantic-gpu/bin/python -B -m captioning download-metrics --model st5 --execute
.venv/rscc-semantic-gpu/bin/python -B -m captioning download-metrics --model bert --execute
```

Destinos existentes são recusados. Confira `st5_model` e `bert_model` na
configuração. Downloads de pesos são explícitos; a geração usa arquivos locais.

## Executar Chg2Cap

Ajuda e planejamento não carregam modelos nem criam saídas:

```bash
python3 -B -m captioning --help
python3 -B -m captioning chg2cap train --output output/novo-treino --hours 4 --dry-run
```

Depois da instalação e preparação, os comandos abaixo executam trabalho.
Cada saída deve ser nova. Os exemplos pressupõem `prepared` configurado para
`data/RSCC/prepared/ebd-novo-v1`; ajuste esse caminho também no `freeze`.

```bash
python3 -B -m captioning chg2cap train --output output/novo-treino --hours 4
python3 -B -m captioning chg2cap validation --run output/novo-treino --output output/nova-validacao --hours 2
python3 -B -m captioning freeze --run output/novo-treino --prepared data/RSCC/prepared/ebd-novo-v1 --validation output/nova-validacao --output output/protocolo-novo.json --generation-limit 551
python3 -B -m captioning chg2cap test --run output/novo-treino --protocol output/protocolo-novo.json --output output/novo-teste --hours 2 --allow-test
```

O treino seleciona o checkpoint pela perda de validação. `freeze` exige
validação completa com os mesmos pesos, dados, geração e código. A geração
livre não recebe tokens da referência. O freeze registra integridade e ordem
das etapas; não comprova que o teste nunca tenha sido consultado.

`--check-only` verifica entradas sem treinar ou gerar, podendo ler imagens e
checkpoint na CPU. `--basic-only` na avaliação omite métricas adicionais.
`--resume caminho/best.pth` retoma treino em uma nova saída; exige
`history.json`, época completa e compatibilidade de dados/código.
Os limites de treino/geração são suaves: uma etapa ou salvamento pode excedê-los.
METEOR não tem timeout próprio neste fluxo; ST5 e BERTScore têm orçamentos
adicionais de 2 h e 1 h por modelo.

## Executar Qwen

O instalador cria `.venv/qwen-rocm` e compartilha bibliotecas ROCm da base
`venv_rocm` por um arquivo `.pth`. Esse ambiente depende do caminho da base;
ao mudar de máquina, recrie ambos. Setup e download requerem `--execute`:

```bash
python3 -B -m captioning qwen setup --dry-run
python3 -B -m captioning qwen setup --execute
python3 -B -m captioning qwen download --execute
python3 -B -m captioning qwen check
python3 -B -m captioning qwen validation --output output/nova-validacao-qwen --dry-run
```

Antes de executar, ajuste os caminhos Chg2Cap da configuração Qwen para as
predições e scores correspondentes: `chg2cap_validation`, `chg2cap_test`,
`chg2cap_validation_scores` e `chg2cap_test_scores`. Eles são usados na comparação;
geração Qwen isolada não exige predições Chg2Cap. Finalize a configuração antes
do freeze, pois ela inteira integra o protocolo.

```bash
python3 -B -m captioning qwen validation --output output/nova-validacao-qwen --seconds 7200
python3 -B -m captioning qwen metrics --evaluation output/nova-validacao-qwen --output output/scores-validacao-qwen --seconds 7200
python3 -B -m captioning qwen freeze --validation output/nova-validacao-qwen --output output/protocolo-qwen.json
python3 -B -m captioning qwen test --protocol output/protocolo-qwen.json --output output/novo-teste-qwen --seconds 7200 --allow-test
python3 -B -m captioning qwen metrics --evaluation output/novo-teste-qwen --output output/scores-teste-qwen --seconds 7200 --allow-test
python3 -B -m captioning qwen compare --evaluation output/novo-teste-qwen --scores output/scores-teste-qwen --output output/comparacao-nova --allow-test
```

O protocolo Qwen fixa revisão dos pesos, prompt, precisão, resolução e parâmetros
de geração. A comparação exige os mesmos IDs/referências e scores íntegros.
Os limites são externos, com até cinco segundos adicionais para encerrar o
processo; nas métricas, o orçamento vale por avaliador.

## Testes e saídas

```bash
venv_rocm/bin/python -B -m unittest discover -s tests -p 'test_captioning*.py' -v
```

Os testes cobrem carregamento e pareamento temporal, vocabulário/splits sem
vazamento, preservação dos tokens, geração com objetos simulados, retomada CPU,
contratos da interface, congelamento de protocolo, métricas sintéticas e uso
em cópia sem dados/pesos. O padrão seleciona a implementação corrente;
os testes históricos preservados localmente têm outros nomes.
Eles não executam treino nem inferência com modelos reais.

Treino grava `config.json`, `history.json`, `result.json`, `best.pth` e
`last.pth`. Avaliação grava predições JSON, `metrics.json` e `comparison.html`;
scores adicionais ficam em `scores/`. Qwen grava `qwen-predictions.json`,
`protocol.json` e `status.json`, com scores/comparação em saídas separadas.
Preserve os manifestos, configurações, hashes, fontes e versões de cada execução.

BLEU/ROUGE, METEOR-NLTK, ST5-SCS e BERTScore medem relações textuais,
sem comprovar acerto visual. Referências sintéticas também podem conter erros.
O teste não deve orientar ajustes; EBD já consultado continua sendo um teste
conhecido. Não misture scores antigos com predições de novas execuções.

A seleção de código foi verificada a partir do índice Git, usando bibliotecas
dos ambientes locais existentes. Isso não certifica instalação pela rede,
portabilidade da GPU ou reprodução dos experimentos concluídos.

## Projeto original

[Changes to Captions: An Attentive Network for Remote Sensing Change Captioning](https://arxiv.org/abs/2304.01091),
Shizhen Chang e Pedram Ghamisi, IARAI, IEEE Transactions on Image Processing.

![Arquitetura Chg2Cap](Figure/Flowchart.png)

```bibtex
@article{chg2cap,
  title={Changes to Captions: An Attentive Network for Remote Sensing Change Captioning},
  author={Chang, Shizhen and Ghamisi, Pedram},
  journal={IEEE Trans. Image Process.},
  doi={10.1109/TIP.2023.3328224},
  year={2023}
}
```

O projeto original utiliza [LEVIR-CC](https://github.com/Chen-Yang-Liu/RSICC)
e [Dubai-CC](https://disi.unitn.it/~melgani/datasets.html).
Código sob a [licença MIT original](LICENSE.txt). Os dados e pesos têm condições
próprias de acesso e redistribuição.
