"""
模型定义：基于 chinese-roberta-wwm-ext 的二分类模型（Human vs AI-Generated）。
在 [CLS] token 上接线性分类头。
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, PreTrainedModel

logger = logging.getLogger(__name__)


class RobertaForAITextDetection(PreTrainedModel):
    """
    BERT 二分类模型：用于检测中文 AI 生成文本。

    架构: RoBERTa + Dropout + Linear(768 → 2)
    """

    config_class = AutoConfig
    base_model_prefix = "roberta"

    def __init__(self, config, roberta: nn.Module | None = None):
        super().__init__(config)
        self.num_labels = config.num_labels

        # 预训练 BERT backbone（可接受外部传入的预训练模型，避免双重内存分配）
        if roberta is not None:
            self.roberta = roberta
        else:
            self.roberta = AutoModel.from_config(config)

        # 分类头
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 调用 post_init 确保 transformers 内部状态完整（如 tied_weights_keys），
        # 并自动初始化分类头权重。
        self.post_init()

        # 兼容新版 transformers（>=4.49）的 from_pretrained 校验
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys: dict = {}
        if not hasattr(self, "_tied_weights_keys"):
            self._tied_weights_keys: list = []

    def _init_weights(self, module: nn.Module) -> None:
        """初始化新增模块的权重，跳过 backbone 内部的模块以保护预训练权重。"""
        if isinstance(module, nn.Linear):
            for child in self.children():
                if child is module:
                    module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                    if module.bias is not None:
                        module.bias.data.zero_()
                    return

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        # BERT encoder
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
            **kwargs,
        )

        # 取 [CLS] token 的 hidden state
        pooled_output = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)

        # 分类头
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # (batch_size, num_labels)

        return {"logits": logits}

    @classmethod
    def from_pretrained_with_head(
        cls,
        model_name_or_path: str,
        num_labels: int = 2,
        dropout_rate: float = 0.1,
    ) -> "RobertaForAITextDetection":
        """
        从预训练模型加载并添加分类头。

        Args:
            model_name_or_path: 预训练模型路径或 HuggingFace ID
            num_labels: 分类类别数
            dropout_rate: 分类头的 dropout 概率
        """
        logger.info(f"加载预训练模型: {model_name_or_path}")

        config = AutoConfig.from_pretrained(model_name_or_path, local_files_only=True)
        config.num_labels = num_labels
        config.hidden_dropout_prob = dropout_rate

        # 先加载预训练 backbone，再传入构造函数，避免同时存在两份 backbone 导致 OOM
        pretrained_backbone = AutoModel.from_pretrained(model_name_or_path, local_files_only=True)
        model = cls(config, roberta=pretrained_backbone)

        logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        return model
