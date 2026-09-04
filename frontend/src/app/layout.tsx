import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import type { ReactNode } from "react";

import { Nav } from "@/components/Nav";

import "./globals.css";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "AIP Dashboard",
  description: "ADAS Intelligence Platform — evidence-backed AEB diagnostics",
};

// Explicit props rather than the generated `LayoutProps` type, so `tsc --noEmit` passes on a
// clean checkout before `next build` has produced .next/types.
export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}>
      <body className="min-h-full flex flex-col bg-bg text-fg">
        <Nav />
        <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-6">{children}</main>
        <footer className="border-t border-line px-4 py-3 text-center text-xs text-muted">
          Engineering assistance, not certified safety tooling. Synthetic evidence is never real-world
          validation.
        </footer>
      </body>
    </html>
  );
}
