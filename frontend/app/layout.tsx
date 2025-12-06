import type { Metadata } from "next";
import "./globals.css";
import "./dynamic-styles.css";

export const metadata: Metadata = {
  title: "BioMLStudio",
  description: "Automated ML for Biomedical Data",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
