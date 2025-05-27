import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import torch
from replay.metrics import OfflineMetrics, Recall, Precision, MAP, NDCG, HitRate, MRR
from replay.metrics.torch_metrics_builder import metrics_to_df
from replay.splitters import LastNSplitter
from replay.data import (
    FeatureHint,
    FeatureInfo,
    FeatureSchema,
    FeatureSource,
    FeatureType,
    Dataset,
)
from replay.models.nn.optimizer_utils import FatOptimizerFactory
from replay.models.nn.sequential.callbacks import (
    ValidationMetricsCallback,
    SparkPredictionCallback,
    PandasPredictionCallback,
    TorchPredictionCallback,
    QueryEmbeddingsPredictionCallback,
)
from replay.models.nn.sequential.postprocessors import RemoveSeenItems
from replay.data.nn import (
    SequenceTokenizer,
    SequentialDataset,
    TensorFeatureSource,
    TensorSchema,
    TensorFeatureInfo
)
from replay.models.nn.sequential import Bert4Rec
from replay.models.nn.sequential.bert4rec import (
    Bert4RecPredictionDataset,
    Bert4RecTrainingDataset,
    Bert4RecValidationDataset,
    Bert4RecPredictionBatch,
    Bert4RecModel
)
import pandas as pd

!pip install rs-datasets

from rs_datasets import MovieLens

movielens = MovieLens("1m")
interactions = movielens.ratings
user_features = movielens.users
item_features = movielens.items

interactions.head()

user_features.head()

item_features.head()

interactions["timestamp"] = interactions["timestamp"].astype("int64")
interactions = interactions.sort_values(by="timestamp")
interactions["timestamp"] = interactions.groupby("user_id").cumcount()
interactions

splitter = LastNSplitter(
    N=1,
    divide_column="user_id",
    query_column="user_id",
    strategy="interactions",
)
raw_test_events, raw_test_gt = splitter.split(interactions)
raw_validation_events, raw_validation_gt = splitter.split(raw_test_events)
raw_train_events = raw_validation_events

def prepare_feature_schema(is_ground_truth: bool) -> FeatureSchema:
    base_features = FeatureSchema(
        [
            FeatureInfo(
                column="user_id",
                feature_hint=FeatureHint.QUERY_ID,
                feature_type=FeatureType.CATEGORICAL,
            ),
            FeatureInfo(
                column="item_id",
                feature_hint=FeatureHint.ITEM_ID,
                feature_type=FeatureType.CATEGORICAL,
            ),
        ]
    )
    if is_ground_truth:
        return base_features
    all_features = base_features + FeatureSchema(
        [
            FeatureInfo(
                column="timestamp",
                feature_type=FeatureType.NUMERICAL,
                feature_hint=FeatureHint.TIMESTAMP,
            ),
        ]
    )
    return all_features

train_dataset = Dataset(
    feature_schema=prepare_feature_schema(is_ground_truth=False),
    interactions=raw_train_events,
    query_features=user_features,
    item_features=item_features,
    check_consistency=True,
    categorical_encoded=False,
)

validation_dataset = Dataset(
    feature_schema=prepare_feature_schema(is_ground_truth=False),
    interactions=raw_validation_events,
    query_features=user_features,
    item_features=item_features,
    check_consistency=True,
    categorical_encoded=False,
)
validation_gt = Dataset(
    feature_schema=prepare_feature_schema(is_ground_truth=True),
    interactions=raw_validation_gt,
    check_consistency=True,
    categorical_encoded=False,
)

test_dataset = Dataset(
    feature_schema=prepare_feature_schema(is_ground_truth=False),
    interactions=raw_test_events,
    query_features=user_features,
    item_features=item_features,
    check_consistency=True,
    categorical_encoded=False,
)
test_gt = Dataset(
    feature_schema=prepare_feature_schema(is_ground_truth=True),
    interactions=raw_test_gt,
    check_consistency=True,
    categorical_encoded=False,
)

ITEM_FEATURE_NAME = "item_id_seq"
tensor_schema = TensorSchema(
    TensorFeatureInfo(
        name=ITEM_FEATURE_NAME,
        is_seq=True,
        feature_type=FeatureType.CATEGORICAL,
        feature_sources=[TensorFeatureSource(FeatureSource.INTERACTIONS, train_dataset.feature_schema.item_id_column)],
        feature_hint=FeatureHint.ITEM_ID,
        embedding_dim=300,
    )
)

tokenizer = SequenceTokenizer(tensor_schema, allow_collect_to_master=True)
tokenizer.fit(train_dataset)
sequential_train_dataset = tokenizer.transform(train_dataset)
sequential_validation_dataset = tokenizer.transform(validation_dataset)
sequential_validation_gt = tokenizer.transform(validation_gt, [tensor_schema.item_id_feature_name])
sequential_validation_dataset, sequential_validation_gt = SequentialDataset.keep_common_query_ids(
    sequential_validation_dataset, sequential_validation_gt
)

test_query_ids = test_gt.query_ids
test_query_ids_np = tokenizer.query_id_encoder.transform(test_query_ids)["user_id"].values
sequential_test_dataset = tokenizer.transform(test_dataset).filter_by_query_id(test_query_ids_np)

print(tokenizer.query_id_encoder.mapping, tokenizer.query_id_encoder.inverse_mapping)
print(tokenizer.item_id_encoder.mapping, tokenizer.item_id_encoder.inverse_mapping)

MAX_SEQ_LEN = 100
BATCH_SIZE = 512
NUM_WORKERS = 4
MAX_EPOCHS = 1
model = Bert4Rec(
    tensor_schema,
    block_count=2,
    head_count=4,
    max_seq_len=MAX_SEQ_LEN,
    hidden_size=300,
    dropout_rate=0.5,
    optimizer_factory=FatOptimizerFactory(learning_rate=0.001),
)
checkpoint_callback = ModelCheckpoint(
    dirpath=".checkpoints",
    save_top_k=1,
    verbose=True,
    # if you use multiple dataloaders, then add the serial number of the dataloader to the suffix of the metric name.
    # For example,"recall@10/dataloader_idx_0"
    monitor="recall@10",
    mode="max",
)
validation_metrics_callback = ValidationMetricsCallback(
    metrics=["map", "ndcg", "recall"],
    ks=[1, 5, 10, 20],
    item_count=train_dataset.item_count,
    postprocessors=[RemoveSeenItems(sequential_validation_dataset)]
)
csv_logger = CSVLogger(save_dir=".logs/train", name="Bert4Rec_example")
trainer = L.Trainer(
    max_epochs=MAX_EPOCHS,
    callbacks=[checkpoint_callback, validation_metrics_callback],
    logger=csv_logger,
)
train_dataloader = DataLoader(
    dataset=Bert4RecTrainingDataset(
        sequential_train_dataset,
        max_sequence_length=MAX_SEQ_LEN,
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)
validation_dataloader = DataLoader(
    dataset=Bert4RecValidationDataset(
        sequential_validation_dataset,
        sequential_validation_gt,
        sequential_train_dataset,
        max_sequence_length=MAX_SEQ_LEN,
    ),
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)
trainer.fit(
    model,
    train_dataloaders=train_dataloader,
    val_dataloaders=validation_dataloader,
)

best_model = Bert4Rec.load_from_checkpoint(checkpoint_callback.best_model_path)

prediction_dataloader = DataLoader(
    dataset=Bert4RecPredictionDataset(
        sequential_test_dataset,
        max_sequence_length=MAX_SEQ_LEN,
    ),
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)
csv_logger = CSVLogger(save_dir=".logs/test", name="Bert4Rec_example")

from replay.utils import get_spark_session
spark_session = get_spark_session()

TOPK = [1, 10, 20, 100]
postprocessors = [RemoveSeenItems(sequential_test_dataset)]
spark_prediction_callback = SparkPredictionCallback(
    spark_session=spark_session,
    top_k=max(TOPK),
    query_column="user_id",
    item_column="item_id",
    rating_column="score",
    postprocessors=postprocessors,
)
pandas_prediction_callback = PandasPredictionCallback(
    top_k=max(TOPK),
    query_column="user_id",
    item_column="item_id",
    rating_column="score",
    postprocessors=postprocessors,
)
torch_prediction_callback = TorchPredictionCallback(
    top_k=max(TOPK),
    postprocessors=postprocessors,
)
query_embeddings_callback = QueryEmbeddingsPredictionCallback()
trainer = L.Trainer(
    callbacks=[
        spark_prediction_callback,
        pandas_prediction_callback,
        torch_prediction_callback,
        query_embeddings_callback,
    ],
    logger=csv_logger,
    inference_mode=True
)
trainer.predict(best_model, dataloaders=prediction_dataloader, return_predictions=False)
spark_res = spark_prediction_callback.get_result()
pandas_res = pandas_prediction_callback.get_result()
torch_user_ids, torch_item_ids, torch_scores = torch_prediction_callback.get_result()
user_embeddings = query_embeddings_callback.get_result()

spark_res.show()

pandas_res

print(torch_user_ids[0], torch_item_ids[0], torch_scores[0])

recommendations = tokenizer.query_and_item_id_encoder.inverse_transform(spark_res)

recommendations.show()

best_model_candidates = Bert4Rec.load_from_checkpoint(checkpoint_callback.best_model_path)

TOPK = 2
CANDIDATES = torch.LongTensor([42, 1337])
postprocessors = [RemoveSeenItems(sequential_test_dataset)]
pandas_prediction_callback = PandasPredictionCallback(
    top_k=TOPK,
    query_column="user_id",
    item_column="item_id",
    rating_column="score",
    postprocessors=postprocessors,
)
trainer = L.Trainer(callbacks=[pandas_prediction_callback], logger=csv_logger, inference_mode=True)
best_model_candidates.candidates_to_score = CANDIDATES
trainer.predict(best_model_candidates, dataloaders=prediction_dataloader, return_predictions=False)

pandas_prediction_callback.get_result()

best_model_candidates.candidates_to_score = None

init_args = {"query_column": "user_id", "rating_column": "score"}

result_metrics = OfflineMetrics(
    [Recall(TOPK), Precision(TOPK), MAP(TOPK), NDCG(TOPK), MRR(TOPK), HitRate(TOPK)], **init_args
)(recommendations.toPandas(), raw_test_gt)

metrics_to_df(result_metrics)

user_embeddings

user_embeddings.shape

device = "cuda" if torch.cuda.is_available() else "cpu"
core_model = Bert4RecModel(
    tensor_schema,
    num_blocks=2,
    num_heads=4,
    max_len=MAX_SEQ_LEN,
    hidden_size=300,
    dropout=0.5
)
core_model.eval()
core_model = core_model.to(device)
# Get first batch of data
data = next(iter(prediction_dataloader))
tensor_map, padding_mask, tokens_mask = data.features, data.padding_mask, data.tokens_mask
# Ensure everything is on the same device
padding_mask = padding_mask.to(device)
tokens_mask = tokens_mask.to(device)
tensor_map["item_id_seq"] = tensor_map["item_id_seq"].to(device)
# Get user embeddings
user_embeddings_batch = core_model.get_query_embeddings(tensor_map, padding_mask, tokens_mask)
user_embeddings_batch

user_embeddings_batch.shape

item_sequence = torch.arange(1, 5).unsqueeze(0)[:, -MAX_SEQ_LEN:]
padding_mask = torch.ones_like(item_sequence, dtype=torch.bool)
tokens_mask = padding_mask.roll(-1, dims=0)
tokens_mask[-1, ...] = 0
sequence_item_count = item_sequence.shape[1]

batch = Bert4RecPredictionBatch(
    query_id=torch.arange(0, item_sequence.shape[0], 1).long(),
    padding_mask=padding_mask,
    features={ITEM_FEATURE_NAME: item_sequence.long()},
    tokens_mask=tokens_mask
)

with torch.no_grad():
    scores = best_model.predict(batch)
scores

torch.topk(scores, k=5).indices

with torch.no_grad():
    scores = best_model.predict(batch, candidates_to_score=CANDIDATES)
scores

from replay.models.nn.sequential.compiled import Bert4RecCompiled

best_model = Bert4Rec.load_from_checkpoint(checkpoint_callback.best_model_path)

opt_model = Bert4RecCompiled.compile(
    model=best_model,  # or checkpoint_callback.best_model_path
    mode="one_query",
)

item_sequence = torch.arange(1, 5).unsqueeze(0)[:, -MAX_SEQ_LEN:]
padding_mask = torch.ones_like(item_sequence, dtype=torch.bool)
tokens_mask = padding_mask.roll(-1, dims=0)
tokens_mask[-1, ...] = 0
sequence_item_count = item_sequence.shape[1]

batch = Bert4RecPredictionBatch(
    query_id=torch.arange(0, item_sequence.shape[0], 1).long(),
    padding_mask=padding_mask,
    features={ITEM_FEATURE_NAME: item_sequence.long()},
    tokens_mask=tokens_mask
)

opt_model.predict(batch).shape

batch = Bert4RecPredictionBatch(
    query_id=torch.arange(0, item_sequence.shape[0], 1).long(),
    padding_mask=padding_mask,
    features={ITEM_FEATURE_NAME: item_sequence.long()},
    tokens_mask=tokens_mask
)

opt_model = Bert4RecCompiled.compile(
    model=best_model,
    mode="one_query",
    num_candidates_to_score=2,
)

opt_model.predict(batch, CANDIDATES).shape