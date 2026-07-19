import type { SceneSegmentationSummary } from "../api/schemas";
import { humanizeToken } from "../display";
import { useI18n, type TranslationKey } from "../i18n";

const groupKeys: Record<string, TranslationKey> = {
  road_surface: "scene.group.roadSurface",
  built_environment: "scene.group.builtEnvironment",
  vegetation: "scene.group.vegetation",
  sky: "scene.group.sky",
  vehicles: "scene.group.vehicles",
  traffic_infrastructure: "scene.group.trafficInfrastructure",
  terrain: "scene.group.terrain",
  water: "scene.group.water",
};

const tagKeys: Record<string, TranslationKey> = {
  urban: "scene.tag.urban",
  road_heavy: "scene.tag.roadHeavy",
  vegetation_sparse: "scene.tag.vegetationSparse",
  vegetation_present: "scene.tag.vegetationPresent",
  open_sky: "scene.tag.openSky",
  built_environment_present: "scene.tag.builtEnvironment",
  traffic_infrastructure_visible: "scene.tag.trafficInfrastructure",
};

function coverage(value: number): string {
  return `${value.toFixed(1)}%`;
}

export function SceneAnalysisCard({ scene }: { scene: SceneSegmentationSummary }) {
  const { t } = useI18n();
  const semanticGroups = scene.semantic_label_names_available ? scene.scene_groups : [];
  const semanticTags = scene.semantic_label_names_available ? scene.scene_tags : [];
  const hasAdditionalWarning = scene.warnings.some((warning) => warning !== "segmentation.semantic_label_names_unavailable");
  return (
    <section className="detail-card scene-analysis-card" aria-labelledby="scene-analysis-heading">
      <header className="scene-analysis-header">
        <div>
          <p className="eyebrow">{t("scene.eyebrow")}</p>
          <h2 id="scene-analysis-heading">{t("scene.title")}</h2>
        </div>
        <span className="status-badge">{t("scene.completed")}</span>
      </header>
      <p className="notice notice--warning">{t("scene.notGeographic")}</p>
      {!scene.semantic_label_names_available ? (
        <p className="notice notice--warning" role="status">{t("scene.genericLabels")}</p>
      ) : null}

      <dl className="scene-provider-meta">
        <div><dt>{t("scene.provider")}</dt><dd>{scene.provider}</dd></div>
        <div><dt>{t("scene.device")}</dt><dd>{scene.device}</dd></div>
        <div><dt>{t("scene.runtime")}</dt><dd>{scene.inference_ms.toFixed(1)} ms</dd></div>
        <div><dt>{t("scene.imageSize")}</dt><dd>{scene.image_width} × {scene.image_height}</dd></div>
      </dl>

      <section aria-labelledby="scene-classes-heading">
        <h3 id="scene-classes-heading">{t("scene.dominantClasses")}</h3>
        {scene.dominant_classes.length ? (
          <ul className="scene-ratio-list">
            {scene.dominant_classes.map((item) => {
              const label = scene.semantic_label_names_available
                ? item.class_name
                : t("scene.classFallback", { id: item.class_id });
              return (
                <li key={item.class_id}>
                  <div><span>{label}</span><strong>{coverage(item.percentage)}</strong></div>
                  <span className="scene-ratio-track" aria-hidden="true"><i style={{ width: `${item.percentage}%` }} /></span>
                </li>
              );
            })}
          </ul>
        ) : <p className="muted">{t("scene.none")}</p>}
      </section>

      {semanticGroups.length ? (
        <section aria-labelledby="scene-groups-heading">
          <h3 id="scene-groups-heading">{t("scene.groups")}</h3>
          <ul className="scene-ratio-list scene-group-list">
            {semanticGroups.map((group) => {
              const key = groupKeys[group.name];
              return <li key={group.name}>
                <div><span>{key ? t(key) : humanizeToken(group.name)}</span><strong>{coverage(group.percentage)}</strong></div>
                <span className="scene-ratio-track" aria-hidden="true"><i style={{ width: `${group.percentage}%` }} /></span>
              </li>;
            })}
          </ul>
        </section>
      ) : null}

      {semanticTags.length ? (
        <section aria-labelledby="scene-tags-heading">
          <h3 id="scene-tags-heading">{t("scene.tags")}</h3>
          <ul className="scene-tag-list">
            {semanticTags.map((tag) => {
              const key = tagKeys[tag.name];
              return <li key={`${tag.name}-${tag.reason}`}>
                <strong>{key ? t(key) : humanizeToken(tag.name)}</strong>
                <span>{t("scene.heuristicStrength", { strength: tag.strength.toFixed(2) })}</span>
                <p>{tag.reason}</p>
              </li>;
            })}
          </ul>
        </section>
      ) : null}

      {hasAdditionalWarning ? <p className="muted small">{t("scene.warnings")}: {t("scene.additionalWarning")}</p> : null}
    </section>
  );
}
