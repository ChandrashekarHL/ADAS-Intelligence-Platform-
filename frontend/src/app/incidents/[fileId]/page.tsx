import { IncidentExplorer } from "@/components/IncidentExplorer";

export default async function Page({ params }: { params: Promise<{ fileId: string }> }) {
  const { fileId } = await params;
  return <IncidentExplorer fileId={fileId} />;
}
