# CarVision

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Version](https://img.shields.io/badge/version-1.0.0-blue)

![Demo](demo.gif)

## Introdução

CarVision é uma ferramenta que detecta automoveis e contabiliza eles que oferece o modo para analisar video e conectar em uma camera real e pegar os dados ao vivo. Desenvolvido como um projeto open source, nosso objetivo é o aprendizado e colaboração com a comunidade, estamos utilizando a yolo.

## Funcionalidades

- Detecção de automoveis
- Contabilidade de automoveis

## Pré-requisitos

Antes de começar, certifique-se de ter as seguintes ferramentas instaladas:

- [python] versão 3.12
- [supervision] versão 0.26.1

## Instalação

Siga as etapas abaixo para configurar o projeto em sua máquina local:

1. Clone o repositório:
    ```bash
    git clone https://github.com/IMNascimento/CarVision.git
    ```
2. Navegue até o diretório do projeto:
    ```bash
    cd CarVision
    ```
3. Crie e ative o ambiente virtual:
    ```bash
    python -m venv venv
    source venv/bin/activate  # Para Linux/MacOS
    .\venv\Scripts\activate  # Para Windows
    ```
4. Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```

## Uso

Após a instalação, você pode iniciar a aplicação com o seguinte comando:

```bash
python cars_counter.py --source video.mp4 --save
```

Aguarde ele termina de fazer analise do video

## Exemplos de Uso
```python
# python cars_counter.py --source 01.mp4 --show --save
```

## Contribuindo

Contribuições são bem-vindas! Por favor, siga as diretrizes em CONTRIBUTING.md para fazer um pull request.

## Licença

Distribuído sob a licença MIT. Veja LICENSE para mais informações.

## Autores

Igor Nascimento - Desenvolvedor Principal - [GitHub](https://github.com/IMNascimento)

## Agradecimentos
YOLO
SUPERVISION
