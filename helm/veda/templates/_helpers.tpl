{{/*
Expand the name of the chart.
*/}}
{{- define "veda.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "veda.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
Common labels.
*/}}
{{- define "veda.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" }}
app.kubernetes.io/name: {{ include "veda.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{/*
Selector labels.
*/}}
{{- define "veda.selectorLabels" -}}
app.kubernetes.io/name: {{ include "veda.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{/*
ServiceAccount name.
*/}}
{{- define "veda.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (include "veda.fullname" .) .Values.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}

{{/*
Derived blob endpoint — auto-points at the bundled MinIO when enabled.
*/}}
{{- define "veda.blobEndpoint" -}}
{{- if .Values.env.VEDA_BLOB_ENDPOINT -}}
{{- .Values.env.VEDA_BLOB_ENDPOINT -}}
{{- else if .Values.minio.enabled -}}
{{- printf "http://%s-minio:9000" (include "veda.fullname" .) -}}
{{- end -}}
{{- end -}}

{{/*
Derived NATS endpoint.
*/}}
{{- define "veda.natsEndpoint" -}}
{{- if .Values.env.VEDA_MQ_ENDPOINT -}}
{{- .Values.env.VEDA_MQ_ENDPOINT -}}
{{- else if .Values.nats.enabled -}}
{{- printf "nats://%s-nats:4222" (include "veda.fullname" .) -}}
{{- end -}}
{{- end -}}
