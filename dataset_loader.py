#!/usr/bin/env python3
"""
Módulo para carregamento e preparação de datasets para fine-tuning.
Suporta múltiplos datasets, configurações personalizadas e formatação flexível.
"""

import logging
from typing import List, Optional, Dict, Any
from datasets import load_dataset, concatenate_datasets, Dataset

logger = logging.getLogger(__name__)


class DatasetConfig:
    """Configuração para um dataset individual."""
    
    def __init__(
        self,
        name: str,
        config: Optional[str] = None,
        instruction_column: str = "problem statement",
        response_column: str = "solution",
        filter_language: Optional[str] = None
    ):
        self.name = name
        self.config = config
        self.instruction_column = instruction_column
        self.response_column = response_column
        self.filter_language = filter_language


class DatasetLoader:
    """Carregador de datasets com suporte a múltiplos datasets e formatação."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def load_single_dataset(
        self,
        dataset_config: DatasetConfig,
        format_template: str = "### Instruction:\n{instruction}\n\n### Response:\n{response}"
    ) -> Optional[Dataset]:
        """
        Carrega e processa um único dataset.
        
        Args:
            dataset_config: Configuração do dataset
            format_template: Template de formatação para os exemplos
            
        Returns:
            Dataset formatado ou None se houver erro
        """
        logger.info(f"Carregando dataset: {dataset_config.name}")
        
        try:
            # Carregar dataset
            if dataset_config.config:
                dataset = load_dataset(dataset_config.name, dataset_config.config)
                logger.info(f"Dataset '{dataset_config.name}' (config: {dataset_config.config}) carregado")
            else:
                dataset = load_dataset(dataset_config.name)
                logger.info(f"Dataset '{dataset_config.name}' carregado")
            
            logger.info(f"Total de exemplos: {len(dataset['train'])}")
            
        except Exception as e:
            logger.error(f"Erro ao carregar dataset '{dataset_config.name}': {e}")
            return None
        
        # Filtrar por linguagem se especificado
        if dataset_config.filter_language:
            original_size = len(dataset["train"])
            dataset = dataset.filter(
                lambda x: x.get("language") == dataset_config.filter_language
            )
            filtered_size = len(dataset["train"])
            logger.info(
                f"Filtro de linguagem '{dataset_config.filter_language}': "
                f"{original_size} -> {filtered_size} exemplos"
            )
        
        train_dataset = dataset["train"]
        
        # Formatar exemplos
        def format_example(example):
            # Tentar encontrar colunas de instrução com fallbacks
            instruction = example.get(
                dataset_config.instruction_column,
                example.get('instruction',
                example.get('prompt',
                example.get('input', '')))
            )
            
            # Tentar encontrar colunas de resposta com fallbacks
            response = example.get(
                dataset_config.response_column,
                example.get('response',
                example.get('output',
                example.get('completion',
                example.get('solution', ''))))
            )
            
            # Aplicar template de formatação
            text = format_template.format(
                instruction=instruction,
                response=response
            ) + self.tokenizer.eos_token
            
            return {"text": text}
        
        logger.info(f"Formatando exemplos do dataset '{dataset_config.name}'...")
        try:
            formatted_dataset = train_dataset.map(
                format_example,
                remove_columns=train_dataset.column_names,
                num_proc=4,
                desc=f"Formatando {dataset_config.name}"
            )
            logger.info(f"✓ Dataset '{dataset_config.name}' processado: {len(formatted_dataset)} exemplos")
            return formatted_dataset
            
        except Exception as e:
            logger.error(f"Erro ao formatar dataset '{dataset_config.name}': {e}")
            return None
    
    def load_multiple_datasets(
        self,
        dataset_configs: List[DatasetConfig],
        format_template: str = "### Instruction:\n{instruction}\n\n### Response:\n{response}"
    ) -> Dataset:
        """
        Carrega e combina múltiplos datasets.
        
        Args:
            dataset_configs: Lista de configurações de datasets
            format_template: Template de formatação para os exemplos
            
        Returns:
            Dataset combinado
            
        Raises:
            ValueError: Se nenhum dataset for carregado com sucesso
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Carregando {len(dataset_configs)} dataset(s)")
        logger.info(f"{'='*60}\n")
        
        all_datasets = []
        
        for idx, config in enumerate(dataset_configs):
            logger.info(f"[{idx+1}/{len(dataset_configs)}] Processando: {config.name}")
            dataset = self.load_single_dataset(config, format_template)
            
            if dataset is not None:
                all_datasets.append(dataset)
            
            logger.info("")  # Linha em branco para separação
        
        # Validar que pelo menos um dataset foi carregado
        if len(all_datasets) == 0:
            raise ValueError("Nenhum dataset foi carregado com sucesso!")
        
        # Combinar datasets se houver mais de um
        if len(all_datasets) == 1:
            final_dataset = all_datasets[0]
            logger.info(f"{'='*60}")
            logger.info(f"Dataset final: {len(final_dataset)} exemplos")
            logger.info(f"{'='*60}\n")
        else:
            logger.info(f"{'='*60}")
            logger.info(f"Combinando {len(all_datasets)} datasets...")
            final_dataset = concatenate_datasets(all_datasets)
            logger.info(f"✓ Dataset combinado: {len(final_dataset)} exemplos totais")
            
            # Mostrar distribuição
            logger.info("\nDistribuição:")
            for idx, ds in enumerate(all_datasets):
                percentage = (len(ds) / len(final_dataset)) * 100
                logger.info(f"  - Dataset {idx+1}: {len(ds)} exemplos ({percentage:.1f}%)")
            
            logger.info(f"{'='*60}\n")
        
        return final_dataset


# Configurações predefinidas para datasets adicionais opcionais
PREDEFINED_DATASETS = {
    'evol-instruct': {
        'name': 'nickrosh/Evol-Instruct-Code-80k-v1',
        'instruction_column': 'instruction',
        'response_column': 'output',
        'description': 'Evol-Instruct-Code-80k-v1 - Dataset de código evolutivo com 80k exemplos'
    },
    'magicoder': {
        'name': 'ise-uiuc/Magicoder-OSS-Instruct-75K',
        'instruction_column': 'problem',
        'response_column': 'solution',
        'description': 'Magicoder-OSS-Instruct-75K - Dataset de instruções OSS com 75k exemplos'
    }
}


def load_dataset_from_args(args, tokenizer) -> Dataset:
    """
    Função de conveniência para carregar dataset a partir de argumentos de linha de comando.
    
    Args:
        args: Argumentos parseados (argparse.Namespace)
        tokenizer: Tokenizer do modelo
        
    Returns:
        Dataset formatado e pronto para treinamento
    """
    loader = DatasetLoader(tokenizer)
    
    # Verificar se estamos usando datasets adicionais opcionais
    additional_datasets = []
    if hasattr(args, 'use_evol_instruct') and args.use_evol_instruct:
        config = PREDEFINED_DATASETS['evol-instruct']
        logger.info(f"✓ Adicionando dataset opcional: {config['description']}")
        additional_datasets.append(config)
    
    if hasattr(args, 'use_magicoder') and args.use_magicoder:
        config = PREDEFINED_DATASETS['magicoder']
        logger.info(f"✓ Adicionando dataset opcional: {config['description']}")
        additional_datasets.append(config)
    
    # Verificar se estamos usando múltiplos datasets
    if hasattr(args, 'dataset_names') and isinstance(args.dataset_names, list):
        # Modo múltiplos datasets
        dataset_configs = []
        
        for idx, name in enumerate(args.dataset_names):
            # Obter configuração específica se fornecida
            config = None
            if hasattr(args, 'dataset_configs') and args.dataset_configs:
                if idx < len(args.dataset_configs):
                    config = args.dataset_configs[idx]
            
            # Obter colunas personalizadas se fornecidas
            instruction_col = "problem statement"
            response_col = "solution"
            if hasattr(args, 'dataset_columns'):
                columns = args.dataset_columns.split(",")
                if len(columns) == 2:
                    instruction_col = columns[0].strip()
                    response_col = columns[1].strip()
            
            dataset_config = DatasetConfig(
                name=name,
                config=config,
                instruction_column=instruction_col,
                response_column=response_col,
                filter_language=getattr(args, 'filter_language', None)
            )
            dataset_configs.append(dataset_config)
        
        # Adicionar datasets opcionais se especificados
        for additional_config in additional_datasets:
            dataset_config = DatasetConfig(
                name=additional_config['name'],
                config=None,
                instruction_column=additional_config['instruction_column'],
                response_column=additional_config['response_column'],
                filter_language=None  # Datasets adicionais não usam filtro de linguagem
            )
            dataset_configs.append(dataset_config)
        
        return loader.load_multiple_datasets(dataset_configs)
    
    else:
        # Modo dataset único (compatibilidade com código antigo)
        dataset_name = getattr(args, 'dataset_name', 'hpcgroup/hpc-instruct')
        
        # Se há datasets adicionais, usar modo múltiplo
        if additional_datasets:
            dataset_configs = []
            
            # Adicionar dataset principal
            instruction_col = "problem statement"
            response_col = "solution"
            if hasattr(args, 'dataset_columns'):
                columns = args.dataset_columns.split(",")
                if len(columns) == 2:
                    instruction_col = columns[0].strip()
                    response_col = columns[1].strip()
            
            dataset_config = DatasetConfig(
                name=dataset_name,
                config=getattr(args, 'dataset_config', None),
                instruction_column=instruction_col,
                response_column=response_col,
                filter_language=getattr(args, 'filter_language', None)
            )
            dataset_configs.append(dataset_config)
            
            # Adicionar datasets opcionais
            for additional_config in additional_datasets:
                dataset_config = DatasetConfig(
                    name=additional_config['name'],
                    config=None,
                    instruction_column=additional_config['instruction_column'],
                    response_column=additional_config['response_column'],
                    filter_language=None
                )
                dataset_configs.append(dataset_config)
            
            return loader.load_multiple_datasets(dataset_configs)
        
        else:
            # Modo dataset único puro
            instruction_col = "problem statement"
            response_col = "solution"
            if hasattr(args, 'dataset_columns'):
                columns = args.dataset_columns.split(",")
                if len(columns) == 2:
                    instruction_col = columns[0].strip()
                    response_col = columns[1].strip()
            
            dataset_config = DatasetConfig(
                name=dataset_name,
                config=getattr(args, 'dataset_config', None),
                instruction_column=instruction_col,
                response_column=response_col,
                filter_language=getattr(args, 'filter_language', None)
            )
            
            return loader.load_single_dataset(dataset_config)
